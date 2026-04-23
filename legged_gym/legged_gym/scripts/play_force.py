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

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

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


    # # For sim
    # env_cfg.terrain.terrain_dict = {"flat": 0.25,
    #                                 "rough": 0.,
    #                                 "discrete": 0.25,
    #                                 "parkour_step": 0.,
    #                                 "slop": 0.25,
    #                                 "demo": 0.,
    #                                 "down": 0.125,
    #                                 "up": 0.125
    #                                 }

    # For sim
    env_cfg.terrain.terrain_dict = {"flat": 1.0,
                                    "rough": 0.,
                                    "discrete": 0.0,
                                    "parkour_step": 0.,
                                    "slop": 0.0,
                                    "demo": 0.,
                                    "down": 0.0,
                                    "up": 0.0
                                    }

    # # For sim
    # env_cfg.terrain.terrain_dict = {"flat": 0.25,
    #                                 "rough": 0.,
    #                                 "discrete": 0.25,
    #                                 "parkour_step": 0.,
    #                                 "slop": 0.25,
    #                                 "demo": 0.,
    #                                 "down": 0.0,
    #                                 "up": 0.0
    #                                 }

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

    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
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
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.domain_rand.max_push_ang_vel = 0.5

    # 0 forward         1 backward       2 sideward     3 spining
    env_cfg.env.force_type = 0

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

    env_cfg.domain_rand.randomize_base_com = True

    env_cfg.domain_rand.max_force = 200
    env_cfg.domain_rand.max_torque = 200

    # env_cfg.domain_rand.randomize_link_mass = True
    # env_cfg.domain_rand.link_mass_range = [0.2, 1.8]

    # Narrow
    # traj_length 2000
    # Average_goal_reward tensor(1.0097, device='cuda:0')
    # Success Rate 0.5404339250493096
    # estimation_vel_loss tensor(0.2396, device='cuda:0', grad_fn=<DivBackward0>)
    # estimation_zmp_loss tensor(nan, device='cuda:0', grad_fn=<DivBackward0>)
    # termination_sum 1028
    # termination_border_sum 93

    # traj_length 2000
    # Average_goal_reward tensor(0.9779, device='cuda:0')
    # Success Rate 0.4467195988299206
    # estimation_vel_loss tensor(0.2597, device='cuda:0', grad_fn=<DivBackward0>)
    # estimation_zmp_loss tensor(nan, device='cuda:0', grad_fn=<DivBackward0>)
    # termination_sum 1393
    # termination_border_sum 1324

    # traj_length 2000
    # Average_goal_reward tensor(0.9464, device='cuda:0')
    # Success Rate 0.5515811301192328
    # estimation_vel_loss tensor(0.2678, device='cuda:0', grad_fn=<DivBackward0>)
    # estimation_zmp_loss tensor(nan, device='cuda:0', grad_fn=<DivBackward0>)
    # in zmp rate tensor(0.8898, device='cuda:0')
    # termination_sum 929
    # termination_border_sum 865

    # Average_goal_reward tensor(0.9758, device='cuda:0')
    # Success Rate 0.5104463437796771
    # estimation_vel_loss tensor(0.2705, device='cuda:0', grad_fn=<DivBackward0>)
    # estimation_zmp_loss tensor(nan, device='cuda:0', grad_fn=<DivBackward0>)
    # in zmp rate tensor(0.8921, device='cuda:0')
    # termination_sum 1106
    # termination_border_sum 1031

    # traj_length 2000
    # Average_goal_reward tensor(0.8634, device='cuda:0')
    # Success Rate 0.5182724252491695
    # estimation_vel_loss tensor(0.2981, device='cuda:0', grad_fn=<DivBackward0>)
    # estimation_zmp_loss tensor(nan, device='cuda:0', grad_fn=<DivBackward0>)
    # in zmp rate tensor(0.8707, device='cuda:0')
    # termination_sum 1107
    # termination_border_sum 1015

    # traj_length 2000
    # Average_goal_reward tensor(0.8270, device='cuda:0')
    # Success Rate 0.49340009103322713
    # estimation_vel_loss tensor(0.3105, device='cuda:0', grad_fn=<DivBackward0>)
    # estimation_zmp_loss tensor(nan, device='cuda:0', grad_fn=<DivBackward0>)
    # in zmp rate tensor(0.8678, device='cuda:0')
    # termination_sum 1197
    # termination_border_sum 1113

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
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)


    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    # termination including out border
    termination_sum = 0

    # termination except out border
    termination_border_sum = 0

    # goal_tracking_loss_sum = 0
    goal_reward_sum = 0

    out_border_sum = 0

    human_like_loss_sum = 0
    
    total_length = env_cfg.terrain.num_rows * env_cfg.terrain.terrain_length
    total_width = env_cfg.terrain.num_cols * env_cfg.terrain.terrain_width

    in_border = torch.ones(env.num_envs, dtype=torch.bool)

    traj_length = int(env.max_episode_length)
    print('traj_length', traj_length)

    if args.record_data:
        data_buf = torch.zeros(env.num_envs, traj_length, 10)

    estimation_vel_loss_sum = 0
    estimation_zmp_loss_sum = 0
    in_zmp_support_sum = 0
    zmp_distance_sum = 0
    # estimator.eval()
    # policy.eval()

    # for i in tqdm(range(traj_length)):
    for i in range(traj_length):
        start_time = time.time()

        # traj_length 2000
        # Average_goal_reward tensor(0.8467, device='cuda:0')
        # Success Rate 0.8805601317957167
        # estimation_vel_loss tensor(0.1802, device='cuda:0', grad_fn=<DivBackward0>)
        # estimation_zmp_loss tensor(0.4696, device='cuda:0', grad_fn=<DivBackward0>)
        # termination_sum 214
        # termination_border_sum 145

        # Average_goal_reward tensor(0.5235, device='cuda:0')
        # Success Rate 0.9254658385093167
        # estimation_vel_loss tensor(0.2016, device='cuda:0', grad_fn=<DivBackward0>)
        # estimation_zmp_loss tensor(0.6540, device='cuda:0', grad_fn=<DivBackward0>)
        # termination_sum 127
        # termination_border_sum 84

        z, vel = estimator(obs.detach()[:, train_cfg.estimator.prop_start - env_cfg.env.prop_hist_len * train_cfg.estimator.prop_dim :train_cfg.estimator.prop_start])
        # vel[:, 3:11] += (2 * torch.rand_like(vel[:, 3:11]) - 1) * 0.1
        # vel += (2 * torch.rand_like(vel) - 1) * 1.0

        latent = torch.cat([z,vel],dim = 1).detach()
        # vel_error = torch.abs(vel[:, :3] - env.base_lin_vel * env.obs_scales.lin_vel).mean()
        # zmp_error = torch.abs(vel[:, 3:11] - env.zmp).mean()
        # ground_truth = obs[:, train_cfg.estimator.priv_start:]
        # vel_error = torch.abs(vel[:, :3] - ground_truth[:, :3]).mean()
        # zmp_error = torch.abs(vel[:, 3:7] - ground_truth[:, 3:7]).mean()
        # zmp_error = torch.abs(vel[:, 3:5] - ground_truth[:, 3:5]).mean()

        # breakpoint()
        # print("vel_error", vel_error)
        # print("zmp_error", zmp_error)
        # if vel_error.item() > 10.0:
        #     vel_error = 0.0
        # if zmp_error.item() > 10.0:
        #     zmp_error = 0.0
        # estimation_vel_loss_sum += vel_error
        # estimation_zmp_loss_sum += zmp_error


        obs = torch.cat((obs[:, :train_cfg.estimator.priv_start], latent), dim = 1)

        # actions = policy(obs.detach(), hist_encoding=True)
        # z, vel = estimator(obs.detach()[:, :train_cfg.estimator.prop_dim])
        # latent = torch.cat([z,vel],dim = 1)
        # obs = torch.cat((obs[:, :train_cfg.estimator.priv_start], latent), dim = 1)

        actions = policy(obs.detach(), hist_encoding=True).detach()


        obs, _, rews, dones, infos = env.step(actions.detach())

        # for key in infos['episode']:
        #     print(key)
        # goal_tracking = infos['episode']['rew_tracking_lin_vel'] + infos['episode']['rew_tracking_ang_vel']
        # print('goal_tracking', goal_tracking)
        base_vel = torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2].unsqueeze(1)), dim=-1)  # 合并线速度和角速度
        # print(base_vel.shape)
        # print(env.commands[:, :3].shape)

        lin_reward = torch.exp(-torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1) * env.cfg.rewards.tracking_sigma)
        ang_reward = torch.exp(-torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2]) * env.cfg.rewards.tracking_sigma)

        goal_reward = torch.mean(lin_reward + 0.5 * ang_reward)
    
        # goal_tracking_loss = torch.nn.MSELoss()(base_vel, env.commands[:, :3])
        # goal_tracking_loss_sum += goal_tracking_loss
        goal_reward_sum += goal_reward
        # print('goal_reward', goal_reward)

        # calculate zmp in support ratio
        in_zmp_support = (env.zmp_distance < 0.4).squeeze(1).float().mean()
        in_zmp_support_sum += in_zmp_support

        zmp_distance_sum += env.zmp_distance.mean()

        termination = env.reset_buf * ~env.time_out_buf

        termination_num = torch.sum(termination).item()

        # num_false = torch.sum(~termination).item()

        termination_sum += termination_num


        # # 将走出边界的 termination 置为 False
        # false_termination = termination & ~in_border

        # # 计算被 false 掉的 termination 的数量
        # out_border_num = torch.sum(false_termination).item()

        # out_border_sum += out_border_num
        # print("out_border", torch.sum(~in_border).item())

        termination[~in_border] = False

        
        # 统计 True 的数量
        termination_border_num = torch.sum(termination).item()

        # num_false = torch.sum(~termination).item()

        termination_border_sum += termination_border_num

        # print('termination_num', termination_num)

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


        # print(stop_time-start_time)
    
    try_sum = 1000 + termination_sum
    fail_sum = termination_border_sum
    success_rate = 1 - fail_sum / try_sum

    print('Average_goal_reward', goal_reward_sum / 2000.0)
    print('Success Rate', success_rate)
    print("zmp_distance_sum", zmp_distance_sum / 2000.0)
    # print('Average_human_like_loss', human_like_loss_sum / 2000.0)
    print("estimation_vel_loss", estimation_vel_loss_sum / 2000.0)
    print("estimation_zmp_loss", estimation_zmp_loss_sum / 2000.0)
    print("in zmp rate", in_zmp_support_sum / 2000)

    print("termination_sum", termination_sum)
    print("termination_border_sum", termination_border_sum)




if __name__ == '__main__':
    args = get_args()
    play(args)
