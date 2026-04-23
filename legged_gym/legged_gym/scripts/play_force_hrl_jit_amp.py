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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from PIL import Image
from legged_gym.utils.helpers import get_load_path as get_load_path_auto
from tqdm import tqdm
import time
import torch.nn.functional as F
import torch.nn as nn
  

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint


def compute_kl_divergence(p_obs, q_demo, eps=1e-3):
    p_mean = p_obs.mean(dim=0)
    p_cov = torch.cov(p_obs.T) + eps * torch.eye(p_obs.shape[1])

    q_mean = q_demo.mean(dim=0)
    q_cov = torch.cov(q_demo.T) + eps * torch.eye(q_demo.shape[1])

    # 使用更稳定的 slogdet
    sign_q, logdet_q = torch.linalg.slogdet(q_cov)
    sign_p, logdet_p = torch.linalg.slogdet(p_cov)

    if sign_q <= 0 or sign_p <= 0:
        print("Warning: non-positive definite covariance matrix, skipping KL")
        return torch.tensor(0.0)

    q_cov_inv = torch.inverse(q_cov)

    mean_diff = (q_mean - p_mean).unsqueeze(0)

    trace_term = torch.trace(q_cov_inv @ p_cov)
    mean_term = mean_diff @ q_cov_inv @ mean_diff.T
    log_det_term = logdet_q - logdet_p
    dim = p_obs.shape[1]

    kl = 0.5 * (trace_term + mean_term.item() - dim + log_det_term)
    return kl



def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    args.task = "h1_mimic_eval" if args.task == "h1_mimic" or args.task == "h1_mimic_amp" else args.task
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.motion.motion_curriculum = False      #False

    env_cfg.env.num_envs = 1000 #2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 6
    env_cfg.terrain.terrain_length = 20       # 20
    env_cfg.terrain.terrain_width = 16         #10 12 14


    # # For demo
    # env_cfg.terrain.terrain_dict = {"flat": 0.1,
    #                                 "rough": 0.1,
    #                                 "discrete": 0.,
    #                                 "parkour_step": 0.,
    #                                 "slop": 0.,
    #                                 "demo": 0.8,
    #                                 "down": 0.,
    #                                 "up": 0.
    #                                 }

    # env_cfg.terrain.terrain_dict = {"flat": 1.0,
    #                                 "rough": 0.,
    #                                 "discrete": 0.,
    #                                 "parkour_step": 0.,
    #                                 "slop": 0.,
    #                                 "demo": 0.,
    #                                 "down": 0.,
    #                                 "up": 0.
    #                                 }

    # For sim
    env_cfg.terrain.terrain_dict = {"flat": 0.25,
                                    "rough": 0.,
                                    "discrete": 0.25,
                                    "parkour_step": 0.,
                                    "slop": 0.25,
                                    "demo": 0.,
                                    "down": 0.,
                                    "up": 0.
                                    }
    # Roughness terrain

    # Parkour terrain (step + slop + roughness )

    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    # default True
    env_cfg.commands.heading_command = False
    # env_cfg.commands.ang_vel_clip = 0.4

    env_cfg.env.randomize_start_yaw = False
    env_cfg.env.rand_yaw_range = 0.2    # 1.2
    env_cfg.env.randomize_start_y = False
    env_cfg.env.rand_y_range = 0.2
    env_cfg.env.randomize_start_pitch = False     # Can consider it!
                                         
    # env_cfg.commands.ranges.lin_vel_x = [0.6, 1.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]


    env_cfg.commands.ranges.lin_vel_x = [-0.6, 2.5] # min max [m/s]
    env_cfg.commands.ranges.lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
    env_cfg.commands.ranges.heading = [-1.6, 1.6]



    # Low velocity
    
    # env_cfg.commands.ranges.lin_vel_x = [-0.6, 1.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]



    # High velocity
    
    # env_cfg.commands.ranges.lin_vel_x = [1.0, 2.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]
            

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5   # 5
    env_cfg.domain_rand.max_push_vel_xy = 1.0   # 
    env_cfg.domain_rand.max_push_ang_vel = 0.5


    env_cfg.noise.noise_scale = 0.0
    env_cfg.noise.noise_scales.dof_pos = 0.02
    env_cfg.noise.noise_scales.dof_vel = 0.20
    env_cfg.noise.noise_scales.ang_vel = 0.50
    env_cfg.noise.noise_scales.imu = 0.2
    env_cfg.noise.noise_scales.gravity = 0.1

    env_cfg.domain_rand.randomize_pd_gain = False  # 启用PD增益随机化
    env_cfg.domain_rand.kp_range = [0.8, 1.2]   # kp增益范围
    env_cfg.domain_rand.kd_range = [0.8, 1.2]   # kd增益范围

    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.added_mass_range = [0, 5]


    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.link_mass_range = [0.2, 1.8]


    env_cfg.domain_rand.max_force = 200
    env_cfg.domain_rand.max_torque = 200

    args.record_data = True


    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # record data
    stop_state_log = 600 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    logger = Logger(env.dt)


    # env.device = 'cpu'

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    # print(train_cfg)
    # ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

  
    # policy = ppo_runner.get_inference_policy(device=env.device)
    # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    termination_sum = 0

    termination_border_sum = 0

    goal_reward_sum = 0

    in_zmp_support_sum = 0

    estimation_vel_loss_sum = 0

    estimation_zmp_loss_sum = 0



    total_changes = 0

    total_flip_changes = 0

    in_border = torch.ones(env.num_envs, dtype=torch.bool)

    total_length = env_cfg.terrain.num_rows * (env_cfg.terrain.terrain_length + 2 * env_cfg.terrain.border_size)
    total_width = env_cfg.terrain.num_cols * (env_cfg.terrain.terrain_width + 2 * env_cfg.terrain.border_size)

    # human_like_loss_sum = 0
    # loco_path = '/home/simon/expressive-humanoid/legged_gym/logs/h1/walk_locomotion_test8_3_no_zmp/traced/walk_locomotion_test8_3_no_zmp_reward-39800-actor_jit.pt'


    # loco_path = '/home/simon/expressive-humanoid/legged_gym/logs/h1/walk_locomotion_test8_3_no_zmp_reward/traced/walk_locomotion_test8_3_no_zmp_reward-39800-actor_jit.pt'
    
    # # test   2   no termination
    # loco_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/2/walk_ours_real_deploy-29800-actor_jit.pt'
    # reco_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/2/walk_recovery_real_deploy-39800-actor_jit.pt'

    # selector_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/2/selector_model_5000.pt'


    # test   3   termination = 100
    # loco_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/3/walk_ours_real_deploy-29800-actor_jit.pt'
    # reco_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/3/walk_recovery_real_deploy-39800-actor_jit.pt'

    # selector_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/3/selector_model_5000.pt'

    # # test   4    termination 50
    # loco_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/2/walk_ours_real_deploy-29800-actor_jit.pt'
    # reco_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/2/walk_recovery_real_deploy-39800-actor_jit.pt'

    # selector_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/2/selector_model_5000.pt'

    # loco_path = '/data1/sixu/selector_policy/test/nips/0.0001_2/goal_tracking-34800-actor_jit.pt'
    # reco_path = '/data1/sixu/selector_policy/test/nips/0.0001_2/recovery-34800-actor_jit.pt'

    loco_path = '/data1/sixu/selector_policy/test/nips/original/goal_tracking-34800-actor_jit.pt'
    reco_path = '/data1/sixu/selector_policy/test/nips/original/recovery-34800-actor_jit.pt'

    # loco_path = '/data1/sixu/HWC_Loco/legged_gym/logs/h1/goal_tracking/traced/goal_tracking-34800-actor_jit.pt'
    # reco_path = '/data1/sixu/HWC_Loco/legged_gym/logs/h1/recovery/traced/recovery-34800-actor_jit.pt'

    # hwc-loco-l
    # selector_path = '/data1/sixu/selector_policy/test/nips/0.0001_2/selector_model_4000.pt'

    # hwc-loco
    selector_path = '/data1/sixu/selector_policy/test/nips/original/selector_model_10000.pt'

    # hwc-loco
    # selector_path = '/home/simon/expressive-humanoid/legged_gym/logs/change_policy/nips/0.0001_2/selector_model_10000.pt'
    # low-freq

    # Average_goal_reward tensor(1.0171, device='cuda:0')
    # Success Rate 0.94362292051756
    # in zmp rate tensor(0.8341, device='cuda:0')
    # total_changes 182.50800828263164
    # termination_sum 82
    # termination_border_sum 61

    # Average_goal_reward tensor(1.0437, device='cuda:0')
    # Success Rate 0.9324817518248175
    # in zmp rate tensor(0.8335, device='cuda:0')
    # total_changes 20.5430000616470352
    # termination_sum 96
    # termination_border_sum 74

    
    # fix 0.2
    # Average_goal_reward tensor(0.9348, device='cuda:0')
    # Success Rate 0.948339483394834
    # in zmp rate tensor(0.8122, device='cuda:0')
    # total_changes 461.10402178391814
    # termination_sum 84
    # termination_border_sum 56

    # fix 0.4
    # Average_goal_reward tensor(1.0231, device='cuda:0')
    # Success Rate 0.9454209065679926
    # in zmp rate tensor(0.8326, device='cuda:0')
    # total_changes 189.6710086381063
    # termination_sum 81
    # termination_border_sum 59

    # low impulse
    # Average_goal_reward tensor(0.9199, device='cuda:0')
    # Success Rate 0.9560336763330215
    # in zmp rate tensor(0.8285, device='cuda:0')
    # total_changes 197.78200913593173
    # total_flip_changes 84.02500365301967
    # termination_sum 69
    # termination_border_sum 47

    # Average_goal_reward tensor(0.9527, device='cuda:0')
    # Success Rate 0.9418819188191881
    # in zmp rate tensor(0.8280, device='cuda:0')
    # total_changes 5.838000254938379
    # total_flip_changes 2.7940001391107216
    # estimation_vel_loss tensor(0.1961, device='cuda:0')
    # estimation_zmp_loss tensor(0.4544, device='cuda:0')
    # termination_sum 84
    # termination_border_sum 63

    # fix 0.2
    # Average_goal_reward tensor(0.8296, device='cuda:0')
    # Success Rate 0.95045197740113
    # in zmp rate tensor(0.8113, device='cuda:0')
    # total_changes 467.6090224478394
    # termination_sum 62
    # termination_border_sum 42
    
    # fix 0.4
    # Average_goal_reward tensor(0.9262, device='cuda:0')
    # Success Rate 0.9580615097856477
    # in zmp rate tensor(0.8273, device='cuda:0')
    # total_changes 208.69700952619314
    # termination_sum 73
    # termination_border_sum 45

    # high impulse
    # Average_goal_reward tensor(0.8789, device='cuda:0')
    # Success Rate 0.6520076481835564
    # in zmp rate tensor(0.8053, device='cuda:0')
    # total_changes 215.08300994709134
    # termination_sum 569
    # termination_border_sum 546

    # hard code 0.2
    # Average_goal_reward tensor(0.8024, device='cuda:0')
    # Success Rate 0.8431372549019608
    # in zmp rate tensor(0.7866, device='cuda:0')
    # total_changes 465.68202187120914
    # termination_sum 224
    # termination_border_sum 192

    # hard code 0.4
    # Average_goal_reward tensor(0.9281, device='cuda:0')
    # Success Rate 0.7584187408491947
    # in zmp rate tensor(0.8124, device='cuda:0')
    # total_changes 214.43800982646644
    # termination_sum 366
    # termination_border_sum 330


    # constant
    # Average_goal_reward tensor(0.9781, device='cuda:0')
    # Success Rate 0.6623544631306597
    # in zmp rate tensor(0.8214, device='cuda:0')
    # total_changes 192.68500869348645
    # total_flip_changes 82.03500354290009
    # estimation_vel_loss tensor(0.1882, device='cuda:0')
    # estimation_zmp_loss tensor(0.4556, device='cuda:0')
    # termination_sum 546
    # termination_border_sum 522

    # Average_goal_reward tensor(1.0090, device='cuda:0')
    # Success Rate 0.6268292682926829
    # in zmp rate tensor(0.8241, device='cuda:0')
    # total_changes 6.387000306509435
    # total_flip_changes 2.935000127297826
    # estimation_vel_loss tensor(0.1897, device='cuda:0')
    # estimation_zmp_loss tensor(0.4614, device='cuda:0')
    # termination_sum 640
    # termination_border_sum 612


    # fix 0.4
    # Average_goal_reward tensor(0.8683, device='cuda:0')
    # Success Rate 0.4734432234432234
    # in zmp rate tensor(0.7914, device='cuda:0')
    # total_changes 232.65901061706245
    # termination_sum 1184
    # termination_border_sum 1150


    # Average_goal_reward tensor(1.0196, device='cuda:0')
    # Success Rate 0.9392824287028518
    # in zmp rate 0.0
    # termination_sum 87
    # termination_border_sum 66


    # Average_goal_reward tensor(1.0182, device='cuda:0')
    # Success Rate 0.9330275229357798
    # in zmp rate 0.0
    # termination_sum 90
    # termination_border_sum 73

    # hwc-loco-l
    # Average_goal_reward tensor(1.0032, device='cuda:0')
    # Success Rate 0.6253813300793167
    # in zmp rate 0.0
    # termination_sum 639
    # termination_border_sum 614

    # hwc-loco
    # Average_goal_reward tensor(1.0532, device='cuda:0')
    # Success Rate 0.6574550128534704
    # in zmp rate 0.0
    # termination_sum 556
    # termination_border_sum 533

    # zmp 0.2
    # Average_goal_reward tensor(0.8975, device='cuda:0')
    # Success Rate 0.7043895747599451
    # in zmp rate 0.0
    # termination_sum 458
    # termination_border_sum 431

    # zmp 0.3
    # Average_goal_reward tensor(0.9546, device='cuda:0')
    # Success Rate 0.6872920825016633
    # in zmp rate 0.0
    # termination_sum 503
    # termination_border_sum 470



    selector_state_dict = torch.load(selector_path, map_location=env.device)
    
    selector_input = env_cfg.env.n_feature + env_cfg.env.n_proprio + env_cfg.env.n_demo + env_cfg.env.n_decoder_out

    selector = SelectorNetwork(selector_input).to(env.device)  # 替换为实际的网络类
    
    selector.load_state_dict(selector_state_dict)


    locomotion_policy = torch.jit.load(loco_path, map_location=env.device)
    recovery_policy =  torch.jit.load(reco_path, map_location=env.device)

    estimator = locomotion_policy.estimator

    z, vel = estimator(obs.detach()[:, train_cfg.estimator.prop_start - env_cfg.env.prop_hist_len * train_cfg.estimator.prop_dim :train_cfg.estimator.prop_start])
    latent = torch.cat([z,vel],dim = 1)

        
    total_kl_div = 0
    traj_length = int(env.max_episode_length)
    print('traj_length', traj_length)

    if args.record_data:
        data_buf = torch.zeros(env.num_envs, traj_length, 10)

    with torch.no_grad():

        for i in tqdm(range(traj_length)):
            start_time = time.time()
            obs_selector = torch.cat([obs[:, : train_cfg.estimator.prop_start + train_cfg.estimator.prop_dim], latent], dim = 1)

            q_values = selector(obs_selector)

            selection = q_values.argmax(dim=1).long()

            locomotion_mask = selection == 0
            recovery_mask = selection == 1

            # # compute the ZMP selection
            # locomotion_mask = (env.zmp_distance < 0.2).squeeze(1)  # 0.1   0.2   0.4
            # recovery_mask = (env.zmp_distance > 0.2).squeeze(1)  # 0.3

            # 统计所有状态切换：当前状态与上一状态不同则计数
            # 公式：(当前为locomotion且上一时刻非) OR (当前非locomotion且上一时刻是)
            if i > 0:
                current_changes = (locomotion_mask & ~last_locomotion_mask) | (~locomotion_mask & last_locomotion_mask)
                
                # 累加本次迭代的切换次数（统计所有样本的切换总数）
                # total_changes += current_changes.sum().item()
                total_changes += current_changes.float().mean().item()
            if i > 1:
                flip_changes = (locomotion_mask & ~last_locomotion_mask & last_last_locomotion_mask) | (~locomotion_mask & last_locomotion_mask & ~last_last_locomotion_mask)
                total_flip_changes += flip_changes.float().mean().item()

            # 更新上一时刻的状态，用于下一循环比较
            if i > 0:
                last_last_locomotion_mask = last_locomotion_mask.clone()
            last_locomotion_mask = locomotion_mask.clone()  # 注意用clone()避免引用问题



                # 0.9584   0.9134
                # 0.9684   0.9345,

            # Average_goal_reward tensor(0.9940, device='cuda:0')
            # Success Rate 0.9242843951985227
            # termination_sum 83
            # termination_border_sum 82

            # Average_goal_reward tensor(0.9734, device='cuda:0')
            # Success Rate 0.9199632014719411
            # termination_sum 87
            # termination_border_sum 87


            # Average_goal_reward tensor(0.7021, device='cuda:0')
            # Success Rate 0.7694610778443114
            # termination_sum 336
            # termination_border_sum 308

            # Average_goal_reward tensor(0.6658, device='cuda:0')
            # Success Rate 0.8153724247226625
            # termination_sum 262
            # termination_border_sum 233

            # Average_goal_reward tensor(0.6161, device='cuda:0')
            # Success Rate 0.8448979591836735
            # termination_sum 225
            # termination_border_sum 190

            # Average_goal_reward tensor(0.7246, device='cuda:0')
            # Success Rate 0.7732035928143712
            # in zmp rate tensor(0.1171, device='cuda:0')

            # calculate zmp in support ratio
            in_zmp_support = (env.zmp_distance < 0.4).squeeze(1).float().mean()
            in_zmp_support_sum += in_zmp_support


            # Average_goal_reward tensor(0.7464, device='cuda:0')
            # Success Rate 0.756578947368421
            # in zmp rate tensor(0.5906, device='cuda:0')
            # termination_sum 368
            # termination_border_sum 333


            # Average_goal_reward tensor(0.7246, device='cuda:0')
            # Success Rate 0.7927858787413661
            # in zmp rate tensor(0.3655, device='cuda:0')
            # termination_sum 303
            # termination_border_sum 270



            # Average_goal_reward tensor(0.7271, device='cuda:0')
            # Success Rate 0.7300916138125441
            # in zmp rate tensor(0.3634, device='cuda:0')
            # termination_sum 419
            # termination_border_sum 383

            # Select action
            if locomotion_mask.any():
                actions[locomotion_mask] = locomotion_policy(obs[locomotion_mask].detach())

            if recovery_mask.any():
                actions[recovery_mask] = recovery_policy(obs[recovery_mask].detach())
                print('recovery_sum', torch.sum(recovery_mask))

            # actions = locomotion_policy(obs.detach())
            # # actions = recovery_policy(obs.detach())

            
            obs, _, rews, dones, infos = env.step(actions.detach())
            # amp_obs = infos["amp_obs"].cpu()  # policy obs
            # amp_obs[:, 0] = 0
            # amp_demo_fetch_batch_size = 100
            # amp_demo_obs = env.fetch_amp_obs_demo(amp_demo_fetch_batch_size).cpu()  # reference obs
            # amp_demo_obs[:, 0] = 0

            # kl_div = compute_kl_divergence(amp_obs, amp_demo_obs)
            # print("KL Divergence (policy || demo):", kl_div)
            # total_kl_div += kl_div

            if locomotion_mask.any():
                z, v = locomotion_policy.estimator(obs.detach()[:, train_cfg.estimator.prop_start - (env_cfg.env.prop_hist_len-1) * train_cfg.estimator.prop_dim :train_cfg.estimator.prop_start + train_cfg.estimator.prop_dim])
                loco_latent = torch.cat([z,v],dim = 1)
                latent[locomotion_mask] = loco_latent[locomotion_mask]

            if recovery_mask.any():
                z, v = recovery_policy.estimator(obs.detach()[:, train_cfg.estimator.prop_start - (env_cfg.env.prop_hist_len-1) * train_cfg.estimator.prop_dim :train_cfg.estimator.prop_start + train_cfg.estimator.prop_dim])
                reco_latent = torch.cat([z,v],dim = 1)
                latent[recovery_mask] = reco_latent[recovery_mask]

            # # priv_explicit = torch.cat((env.base_lin_vel * env.obs_scales.lin_vel, env.zmp, env.base_height.unsqueeze(1)), dim=-1)
            vel_error = torch.abs(latent[:, 16:19] - env.base_lin_vel * env.obs_scales.lin_vel).mean()
            zmp_error = torch.abs(latent[:, 19:27] - env.zmp).mean()
            print("vel_error", vel_error)
            print("zmp_error", zmp_error)
            if vel_error.item() > 10.0:
                vel_error = 0.0
            if zmp_error.item() > 10.0:
                zmp_error = 0.0
            estimation_vel_loss_sum += vel_error
            estimation_zmp_loss_sum += zmp_error

            lin_reward = torch.exp(-torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1) * env.cfg.rewards.tracking_sigma)
            ang_reward = torch.exp(-torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2]) * env.cfg.rewards.tracking_sigma)

            goal_reward = torch.mean(lin_reward + 0.5 * ang_reward)
        
            # goal_tracking_loss = torch.nn.MSELoss()(base_vel, env.commands[:, :3])
            # goal_tracking_loss_sum += goal_tracking_loss
            goal_reward_sum += goal_reward
            print('goal_reward', goal_reward)

            termination = env.reset_buf * ~env.time_out_buf

            termination_num = torch.sum(termination).item()

            # num_false = torch.sum(~termination).item()

            termination_sum += termination_num


            # # 将走出边界的 termination 置为 False
            # false_termination = termination & ~in_border

            # # 计算被 false 掉的 termination 的数量
            # out_border_num = torch.sum(false_termination).item()

            # out_border_sum += out_border_num
            print("out_border", torch.sum(~in_border).item())

            termination[~in_border] = False

            
            # 统计 True 的数量
            termination_border_num = torch.sum(termination).item()

            # num_false = torch.sum(~termination).item()

            termination_border_sum += termination_border_num

            print('termination_num', termination_num)

            # success_rate =  num_false / (num_false + num_true)

            # print(termination)

            # print('amp_obs', env.extras["amp_obs"].shape)
            # # amp_obs = env.extras["amp_obs"]
            # amp_obs = infos["amp_obs"]

            # amp_demo_fetch_batch_size = 1000
            # amp_demo_obs = env.fetch_amp_obs_demo(amp_demo_fetch_batch_size)

            # human_like_loss = torch.nn.MSELoss()(amp_obs, amp_demo_obs)
            # human_like_loss_sum += human_like_loss

            # print('human_like_loss', human_like_loss)
            # # print(demo_obs.shape)

            in_border = ((0 < env.root_states[:, 0]) & (env.root_states[:, 0] < total_length)) & ((0 < env.root_states[:, 1]) & (env.root_states[:, 1] < total_width))
            

            if args.record_data:
                data_buf[env.lookat_id, i, 0] = env.commands[env.lookat_id, 0]    # command_x
                data_buf[env.lookat_id, i, 1] = env.commands[env.lookat_id, 1]    # command_y
                data_buf[env.lookat_id, i, 2] = env.commands[env.lookat_id, 2]    # command_yaw
                data_buf[env.lookat_id, i, 3] = env.base_lin_vel[env.lookat_id, 0]   # base x vel
                data_buf[env.lookat_id, i, 4] = env.base_lin_vel[env.lookat_id, 1]   # base y vel
                data_buf[env.lookat_id, i, 5] = env.base_lin_vel[env.lookat_id, 2]   # base z vel
                data_buf[env.lookat_id, i, 6] = env.base_ang_vel[env.lookat_id, 2]   # base yaw vel
                data_buf[env.lookat_id, i, 7:9] = env.contact_forces[env.lookat_id, env.feet_indices, 2]    # contact_force
                # data_buf[env.lookat_id, i, 9] = env.zmp[env.lookat_id]               # zmp feature

            if args.web:
                web_viewer.render(fetch_results=True,
                            step_graphics=True,
                            render_all_camera_sensors=True,
                            wait_for_page_load=True)

            # Interaction
            if env.button_pressed:
                print(
                    # f"env_id: {env.lookat_id:<{5}}"
                    #   f"motion file: {env._motion_lib.get_motion_files([env._motion_ids[env.lookat_id]])[0].split('/')[-1].split('.')[0]:<{10}}"
                    f"vx: {env.commands[env.lookat_id, 0]:<{8}.2f}"
                    f"vy: {env.commands[env.lookat_id, 1]:<{8}.2f}"
                    f"d_yaw: {env.commands[env.lookat_id, 3]:<{8}.2f}"
                    #   f"descript# time.sleep(env.dt)
                )

            # To record and visualize the data
            if i < stop_state_log:
                logger.log_states(
                    {
                        # 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        # 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        # 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        # 'dof_torque': env.torques[robot_index, joint_index].item(),
                        # 'command_x': env.commands[robot_index, 0].item(),
                        'command_x':env.commands[env.lookat_id, 0].item(),
                        'command_y': env.commands[env.lookat_id, 1].item(),
                        'command_yaw': env.commands[env.lookat_id, 2].item(),
                        # 'command_ori': env.target_ori[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[env.lookat_id, 0].item(),
                        'base_vel_y': env.base_lin_vel[env.lookat_id, 1].item(),
                        'base_vel_z': env.base_lin_vel[env.lookat_id, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[env.lookat_id, 2].item(),
                        # 'base_yaw': env.base_orn_rp[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[env.lookat_id, env.feet_indices, 2].cpu().numpy(),
                        # 'zmp_measure': env.zmp[env.lookat_id].item(),
                        # 'zmp_estimated': vel[env.lookat_id, 3].item()
                        # 'zmp_distance': vel[env.lookat_id, 3].item()
                    }
                )

            elif i==stop_state_log:
                stop_state_log = stop_state_log + i
                # logger.plot_states()

            # time.sleep(env.dt/4)
            stop_time = time.time()


            print(stop_time-start_time)
        
    try_sum = 1000 + termination_sum
    fail_sum = termination_border_sum
    success_rate = 1 - fail_sum / try_sum


    print('Average_goal_reward', goal_reward_sum / 2000.0)
    print('Success Rate', success_rate)
    print("in zmp rate", in_zmp_support_sum / 2000)
    print("kl_div", total_kl_div / 2000.0)
    # print('Average_human_like_loss', human_like_loss_sum / 2000.0)
    print("total_changes", total_changes)
    print("total_flip_changes", total_flip_changes)
    print("estimation_vel_loss", estimation_vel_loss_sum / 2000.0)
    print("estimation_zmp_loss", estimation_zmp_loss_sum / 2000.0)
    
    print("termination_sum", termination_sum)
    print("termination_border_sum", termination_border_sum)


        
class SelectorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=2, dropout_prob=0.1):
        super(SelectorNetwork, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)  # 输出 Q 值，每个动作一个
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)


    def forward(self, x):
        # 前向传播
        x = F.relu(self.fc1(x))  # 第一层，带 LayerNorm 和 ReLU
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # 第二层，带 LayerNorm 和 ReLU
        x = F.relu(self.fc3(x))  # 第三层，带 LayerNorm 和 ReLU
        x = self.fc4(x)                    # 输出层，不加激活函数
        return x


if __name__ == '__main__':
    args = get_args()
    play(args)