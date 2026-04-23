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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage, ReplayBuffer
import wandb
from rsl_rl.utils import unpad_trajectories
import time

class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class PPOMimic:
    def __init__(self,
                 env, 
                 actor_critic,
                 estimator,
                 estimator_paras,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 **kwargs
                 ):

        self.env = env
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.n_demo = estimator_paras["n_demo"]
        self.priv_latent_dim = estimator_paras["priv_latent_dim"]
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.est_start = estimator_paras["priv_start"]
        self.num_prop = estimator_paras["prop_dim"]
        self.prop_start = estimator_paras["prop_start"]
        self.history_len = estimator_paras["history_len"]
        self.future_horizon = estimator_paras.get("future_horizon", 1)

        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Constrained recovery with ZMP cost
        self.use_zmp_cost = kwargs.get("use_zmp_cost", False)
        self.zmp_cost_limit = kwargs.get("zmp_cost_limit", 0.0)
        self.zmp_lambda = torch.tensor(kwargs.get("zmp_lambda_init", 0.0), device=self.device)
        self.zmp_lambda_lr = kwargs.get("zmp_lambda_lr", 1.0e-2)
        self.zmp_lambda_max = kwargs.get("zmp_lambda_max", 20.0)
        self.zmp_cost_value_loss_coef = kwargs.get("zmp_cost_value_loss_coef", 1.0)
        self.normalize_cost_advantages = kwargs.get("normalize_cost_advantages", False)
        self.last_mean_rollout_cost = 0.0
        self._rollout_cost_sum = 0.0
        self._rollout_cost_count = 0


    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        # 这里用预测的linear velocity来训练
        if self.train_with_estimated_states:
            obs_est = obs.clone()

            # For VAE + Improved KL
            hist_obs = obs_est[:, self.prop_start - self.history_len * self.num_prop : self.prop_start]
            current_obs = obs_est[:, self.prop_start : self.prop_start + self.num_prop]
            z, labels = self.estimator.sample(hist_obs, current_obs)
            latent = torch.cat([z, labels], dim = 1)
            obs_est = torch.cat([obs_est[:, : self.est_start], latent], dim = 1)
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        if self.use_zmp_cost:
            self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        else:
            self.transition.cost_values = torch.zeros_like(self.transition.values)
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # self.transition.observations = obs_est
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, next_obs, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        if self.use_zmp_cost:
            zmp_cost = infos.get("zmp_cost", torch.zeros_like(rewards))
            if not isinstance(zmp_cost, torch.Tensor):
                zmp_cost = torch.tensor(zmp_cost, device=self.device, dtype=rewards.dtype)
            zmp_cost = zmp_cost.to(self.device).view(-1)
            self.transition.costs = zmp_cost.clone()
            self._rollout_cost_sum += zmp_cost.sum().item()
            self._rollout_cost_count += zmp_cost.numel()
        else:
            self.transition.costs = torch.zeros_like(rewards)
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            if self.use_zmp_cost:
                self.transition.costs += self.gamma * torch.squeeze(self.transition.cost_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        if self.use_zmp_cost:
            last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
            self.storage.compute_cost_returns(
                last_cost_values,
                self.gamma,
                self.lam,
                normalize_advantages=self.normalize_cost_advantages,
            )
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        num_sub_steps = 1

        mean_estimator_loss = 0
        mean_recon_loss = 0.
        mean_predict_loss = 0.

        mean_discriminator_loss = 0
        mean_discriminator_acc = 0
        mean_priv_reg_loss = 0
        mean_cost_value_loss = 0.
        mean_cost_surrogate_loss = 0.

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for sample in generator:
                obs_batch, next_obs_batch, critic_obs_batch, actions_batch, dones_batch, target_values_batch, advantages_batch, returns_batch, \
                target_cost_values_batch, cost_advantages_batch, cost_returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample

                obs_est_batch = obs_batch.clone()

                # For VAE + Improved KL
                hist_obs_batch = obs_est_batch[:, self.prop_start - self.history_len * self.num_prop : self.prop_start]
                current_obs_batch = obs_est_batch[:, self.prop_start : self.prop_start + self.num_prop]
                z, labels = self.estimator.sample(hist_obs_batch, current_obs_batch)
                latent = torch.cat([z, labels], dim = 1).detach()

                # For Mimic
                # z, vel = self.estimator.sample(obs_est_batch[:, self.prop_start - (self.history_len - 1) * self.num_prop : self.prop_start + self.num_prop])
                # latent = torch.cat([z, vel], dim = 1).detach()

                # # For DreamWaQ
                # z, vel = self.estimator.sample(obs_est_batch[:, self.prop_start: self.prop_start+self.num_prop])
                # latent = torch.cat([z, vel], dim = 1).detach()

                # heights = obs_est_batch[: -132:] 
                obs_est_batch = torch.cat([obs_est_batch[:, : self.est_start], latent], dim = 1)

                self.actor_critic.act(obs_est_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # match distribution dimension

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                if self.use_zmp_cost:
                    cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                else:
                    cost_value_batch = torch.zeros_like(value_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                
                # Estimator
                for i in range(num_sub_steps):
                    future_obs_batch = next_obs_batch[:, self.prop_start : self.prop_start + self.num_prop]
                    future_label_batch = next_obs_batch[:, self.est_start  :  self.est_start + self.priv_states_dim]

                    # For VAE + Improved KL
                    loss_dict = self.estimator.loss_fn(
                        hist_obs_batch,
                        current_obs_batch,
                        future_obs_batch,
                        future_label_batch,
                        dones=dones_batch,
                        kld_weight=1.0,
                    )


                    # For Mimic
                    # loss_dict = self.estimator.loss_fn(obs_batch[:, self.prop_start- (self.history_len - 1) * self.num_prop : self.prop_start + self.num_prop], priv_obs_batch, next_vel_zmp_batch, 1.0)
                    

                    # # For DreamWaQ
                    # loss_dict = self.estimator.loss_fn(obs_batch[:, self.prop_start : self.prop_start + self.num_prop], next_obs_batch, next_vel_zmp_batch, 1.0)
                    

                    estimator_loss = torch.mean(loss_dict['loss'])
                    recon_loss = torch.mean(loss_dict['recons_loss'])
                    predict_loss = torch.mean(loss_dict['label_loss'])
                    
                    
                    self.estimator_optimizer.zero_grad()
                    estimator_loss.backward()
                    nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                    self.estimator_optimizer.step()

                    mean_estimator_loss += estimator_loss.item()
                    mean_recon_loss += recon_loss.item()
                    mean_predict_loss += predict_loss.item()
                

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_zmp_cost:
                    cost_surrogate = torch.squeeze(cost_advantages_batch) * ratio
                    cost_surrogate_clipped = torch.squeeze(cost_advantages_batch) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    cost_surrogate_loss = torch.max(cost_surrogate, cost_surrogate_clipped).mean()
                else:
                    cost_surrogate_loss = torch.zeros((), device=self.device)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                if self.use_zmp_cost:
                    if self.use_clipped_value_loss:
                        cost_value_clipped = target_cost_values_batch + (cost_value_batch - target_cost_values_batch).clamp(
                            -self.clip_param, self.clip_param
                        )
                        cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                        cost_value_losses_clipped = (cost_value_clipped - cost_returns_batch).pow(2)
                        cost_value_loss = torch.max(cost_value_losses, cost_value_losses_clipped).mean()
                    else:
                        cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()
                else:
                    cost_value_loss = torch.zeros((), device=self.device)

                loss = surrogate_loss + \
                       self.zmp_lambda.detach() * cost_surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean()
                if self.use_zmp_cost:
                    loss = loss + self.zmp_cost_value_loss_coef * cost_value_loss
                    #    priv_reg_coef * priv_reg_loss

                # loss = self.teacher_alpha * imitation_loss + (1 - self.teacher_alpha) * loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_cost_surrogate_loss += cost_surrogate_loss.item()
                

                # if self.env.train_estimator == True:
                #     mean_estimator_loss += estimator_loss.item()
                #     mean_recon_loss += recon_loss.item()
                #     mean_predict_loss += predict_loss.item()
                # mean_priv_reg_loss += priv_reg_loss.item()
                # mean_discriminator_loss += 0
                # mean_discriminator_acc += 0
            

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= (num_updates * num_sub_steps)
        mean_recon_loss /= (num_updates * num_sub_steps)
        mean_predict_loss /= (num_updates * num_sub_steps)
        mean_cost_value_loss /= num_updates
        mean_cost_surrogate_loss /= num_updates

        if self.use_zmp_cost:
            self.last_mean_rollout_cost = self._rollout_cost_sum / max(self._rollout_cost_count, 1)
            self.zmp_lambda = torch.clamp(
                self.zmp_lambda + self.zmp_lambda_lr * (self.last_mean_rollout_cost - self.zmp_cost_limit),
                min=0.0,
                max=self.zmp_lambda_max,
            )
        else:
            self.last_mean_rollout_cost = 0.0
        self._rollout_cost_sum = 0.0
        self._rollout_cost_count = 0

        self.storage.clear()
        self.update_counter()
        # return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss, priv_reg_coef
        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimator_loss,
            mean_recon_loss,
            mean_predict_loss,
            mean_cost_value_loss,
            mean_cost_surrogate_loss,
            self.last_mean_rollout_cost,
            float(self.zmp_lambda.item()),
        )

    def update_counter(self):
        self.counter += 1
    
    def calc_amp_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self.amp_discriminator(amp_obs)
            # prob = 1 / (1 + torch.exp(-disc_logits)) 
            # disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))

            disc_r = torch.clamp(1 - (1/4) * torch.square(disc_logits - 1), min=0)

        return disc_r
    
    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward
