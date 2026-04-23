# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from .checkpoint_video_recorder import CheckpointVideoRecorder
import sys
from copy import copy, deepcopy
import warnings
from rsl_rl.utils.utils import Normalizer

class OnPolicyRunnerMimicAMP:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.device = device
        self.env = env
        self.args = kwargs.get("args")

        policy_class = eval(self.cfg["policy_class_name"])
        actor_critic = policy_class(self.env.cfg.env.n_proprio, self.env.cfg.env.n_demo,
                                                      self.env.num_privileged_obs,
                                                      self.env.cfg.env.n_decoder_out,
                                                      self.env.cfg.env.n_priv,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg).to(self.device)
        # estimator = Estimator(input_dim=env.cfg.env.n_proprio * 3, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        estimator = Estimator(
            num_obs=env.cfg.env.n_proprio,
            num_history=self.env.cfg.env.prop_hist_len,
            num_latent=16,
            num_labels=env.cfg.env.n_priv,
            activation='elu',
            decoder_hidden_dims=self.estimator_cfg["decoder_hidden_dims"],
            encoder_hidden_dims=self.estimator_cfg["encoder_hidden_dims"],
        ).to(self.device)

        # AMP
        self.amp_cfg = train_cfg["amp"]
        amp_normalizer = None#Normalizer(self.amp_cfg['amp_input_dim'])
        amp_discriminator = AMPDiscriminator(
            self.amp_cfg['amp_input_dim'],
            self.amp_cfg['amp_disc_hidden_dims'], device,).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(self.env, 
                                  actor_critic, 
                                  estimator, self.estimator_cfg, 
                                  amp_discriminator, amp_normalizer, self.amp_cfg,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
        )

        self.learn = self.learn_RL
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.checkpoint_video_recorder = CheckpointVideoRecorder(self)
        

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.

        mean_estimator_loss = 0.
        mean_recon_loss = 0.
        mean_predict_loss = 0.

        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        # print("This obs", obs.shape)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        task_rew_buf = deque(maxlen=100)
        amp_rew_no_scale_buf = deque(maxlen=100)
        amp_rew_scaled_buf = deque(maxlen=100)
        cur_task_rew_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_rew_no_scale_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_rew_scaled_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        amp_rew_norm = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)


        kld_weight = 0.5

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0
            amp_rew_norm[:] = 0
            if it < 10000:
                kld_weight = 0.5
            else:
                kld_weight = 1.0

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew, task_rew, amp_rew_no_scale, amp_rew_scaled = self.alg.process_env_step(obs, rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1
                        amp_rew_norm += amp_rew_no_scale / 0.02
                        # AMP
                        cur_task_rew_sum += task_rew
                        cur_amp_rew_no_scale_sum += amp_rew_no_scale
                        cur_amp_rew_scaled_sum += amp_rew_scaled

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # AMP
                        task_rew_buf.extend(cur_task_rew_sum[new_ids][:, 0].cpu().numpy().tolist())
                        amp_rew_no_scale_buf.extend(cur_amp_rew_no_scale_sum[new_ids][:, 0].cpu().numpy().tolist())
                        amp_rew_scaled_buf.extend(cur_amp_rew_scaled_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_task_rew_sum[new_ids] = 0
                        cur_amp_rew_no_scale_sum[new_ids] = 0
                        cur_amp_rew_scaled_sum[new_ids] = 0
                amp_rew_norm /= self.num_steps_per_env
                mean_amp_rew_norm = torch.mean(amp_rew_norm)
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            self.alg.update_amp_demos()
            # mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_disc_loss, mean_disc_acc, mean_priv_reg_loss, priv_reg_coef, \
            mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_recon_loss, mean_predict_loss, mean_disc_loss, mean_disc_acc, \
                 mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, entropy_coef  = self.alg.update(kld_weight)

            # if hist_encoding:
            #     print("Updating dagger...")
            #     mean_hist_latent_loss = self.alg.update_dagger()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it < 2500:
                if it % (10*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 30000:
                if it % (10*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))


    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # wandb_dict['Episode_rew/' + key] = value
                if "tracking" in key:
                    wandb_dict['Episode_rew_tracking/' + key] = value
                elif key.startswith("reset_"):
                    wandb_dict['Episode_reset/' + key] = value
                elif "curriculum" in key:
                    wandb_dict['Episode_curriculum/' + key] = value
                elif "terrain_level" in key:
                    wandb_dict['Episode_terrain_level/' + key] = value
                else:
                    wandb_dict['Episode_rew_regularization/' + key] = value
                
                    
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        wandb_dict['Loss/value_func'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        # wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        # wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']

        wandb_dict['Adaptation/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Adaptation/reconstruction'] = locs['mean_recon_loss']
        wandb_dict['Adaptation/prediction'] = locs['mean_predict_loss']
        # wandb_dict['Adaptation/hist_latent_loss'] = locs['mean_hist_latent_loss']
        # wandb_dict['Adaptation/priv_reg_loss'] = locs['mean_priv_reg_loss']
        # wandb_dict['Adaptation/priv_ref_lambda'] = locs['priv_reg_coef']

        wandb_dict['AMP/loss'] = locs['mean_amp_loss']
        wandb_dict['AMP/grad_pen_loss'] = locs['mean_grad_pen_loss']
        wandb_dict['AMP/policy_pred'] = locs['mean_policy_pred']
        wandb_dict['AMP/expert_pred'] = locs['mean_expert_pred']
        wandb_dict['AMP/amp_rew_norm'] = locs['mean_amp_rew_norm']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            # wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = statistics.mean(locs['task_rew_buf'])
            wandb_dict['Train/mean_reward_amp_no_scale'] = statistics.mean(locs['amp_rew_no_scale_buf'])
            wandb_dict['Train/mean_reward_amp_scaled'] = statistics.mean(locs['amp_rew_scaled_buf'])
            # wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                          f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['task_rew_buf']):.2f}\n"""
                          f"""{'Mean reward (amp no scale):':>{pad}} {statistics.mean(locs['amp_rew_no_scale_buf']):.2f}\n"""
                          f"""{'Mean reward (amp scaled):':>{pad}} {statistics.mean(locs['amp_rew_scaled_buf']):.2f}\n"""
                        #   f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                        #   f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)
        self.checkpoint_video_recorder.record(path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        model_state_dict = loaded_dict['model_state_dict']
        if not any(key.startswith("critic_cost.") for key in model_state_dict.keys()):
            for key, value in list(model_state_dict.items()):
                if key.startswith("critic."):
                    model_state_dict["critic_cost." + key[len("critic."):]] = value.clone()
        self.alg.actor_critic.load_state_dict(model_state_dict, strict=False)
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    def get_estimator_inference_policy(self, device=None):
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference
    
    def get_disc_inference_policy(self, device=None):
        self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.discriminator.to(device)
        return self.alg.discriminator.inference
