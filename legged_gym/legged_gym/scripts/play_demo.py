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

def frequency_encoding(x, F):
    """
    频率编码函数
    :param x: 输入的一维张量
    :param F: 正弦和余弦函数的数量
    :return: 频率编码后的张量
    """
    encoding = []
    for i in range(F):
        freq = 2 ** i
        encoding.append(torch.sin(freq * torch.pi * x))
        encoding.append(torch.cos(freq * torch.pi * x))
    return torch.cat(encoding, dim=-1)


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
    if "g1" in args.task:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "g1", args.exptid)
    elif "h1" in args.task:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "h1", args.exptid)
    else:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.motion.motion_curriculum = False      #False

    env_cfg.env.num_envs = 1 #2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.terrain_length = 20       # 20
    env_cfg.terrain.terrain_width = 16         #10 12 14
    env_cfg.terrain.border_size = 5

    env_cfg.env.extreme_flag = False
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
    # env_cfg.terrain.terrain_dict = {"flat": 0.2,
    #                                 "rough": 0.2,
    #                                 "discrete": 0.2,
    #                                 "parkour_step": 0.,
    #                                 "slop": 0.2,
    #                                 "demo": 0.0,
    #                                 "down": 0.2,
    #                                 "up": 0.2
    #                                 }
    
    # # # For sim
    # env_cfg.terrain.terrain_dict = {"flat": 0.2,
    #                                 "rough": 0.,
    #                                 "discrete": 0.2,
    #                                 "parkour_step": 0.,
    #                                 "slop": 0.2,
    #                                 "demo": 0.0,
    #                                 "down": 0.2,
    #                                 "up": 0.2
    #                                 }


    # For sim
    env_cfg.terrain.terrain_dict = {"flat": 1.0,
                                    "rough": 0.,
                                    "discrete": 0.,
                                    "parkour_step": 0.,
                                    "slop": 0.,
                                    "demo": 0.0,
                                    "down": 0.0,
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
    env_cfg.env.randomize_start_pos = True

                                         
    # env_cfg.commands.ranges.lin_vel_x = [2.5, 2.5] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]


    env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8] # min max [m/s]
    env_cfg.commands.ranges.lin_vel_y = [-0.0, 0.0]#[0.15, 0.6]   # min max [m/s]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
    env_cfg.commands.ranges.heading = [-1.6, 1.6]

    # env_cfg.commands.ranges.lin_vel_x = [-0.6, 2.5] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]

    # env_cfg.commands.ranges.lin_vel_x = [ -0.6, 2.5] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]


    # # # # # Stand 
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]


    # Low velocity
    
    # env_cfg.commands.ranges.lin_vel_x = [-0.6, 1.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]


    # # High velocity
    
    # env_cfg.commands.ranges.lin_vel_x = [1.0, 2.0] # min max [m/s]
    # env_cfg.commands.ranges.lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
    # env_cfg.commands.ranges.heading = [-1.6, 1.6]
            
    # env_cfg.asset.terminate_after_contacts_on = ["pelvis"]

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 4
    env_cfg.domain_rand.max_push_vel_xy = 3.0
    env_cfg.domain_rand.max_push_ang_vel = 1.0
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False # False
    
    env_cfg.domain_rand.friction_range = [2.0, 2.0]

    args.record_data = False


    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # record data
    stop_state_log = 600 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    logger = Logger(env.dt)

    if args.record_frame:
        env.enable_viewer_sync = False
        paths = []
        for i in range(env.num_envs):
            frame_name = str(i) + "-"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos_retarget/{frame_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            paths.append(path)

    # env.device = 'cpu'

    if args.web:
        web_viewer.setup(env)
    
    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy = torch.jit.load(path, map_location=env.device)
    else:
        # load policy
        train_cfg.runner.resume = True
        # print(train_cfg)
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

        policy = ppo_runner.get_inference_policy(device=env.device)
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)



    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)


    termination_sum = 0


    goal_tracking_loss_sum = 0


    human_like_loss_sum = 0


    total_length = env_cfg.terrain.num_rows * (env_cfg.terrain.terrain_length + 2 * env_cfg.terrain.border_size)
    total_width = env_cfg.terrain.num_cols * (env_cfg.terrain.terrain_width + 2 * env_cfg.terrain.border_size)

    all_extrem_data = []

    traj_length = int(env.max_episode_length)
    print('traj_length', traj_length)

    in_border = torch.ones(env.num_envs, dtype=torch.bool)

    if args.record_data:
        data_buf = torch.zeros(env.num_envs, traj_length, 10)

    for i in tqdm(range(traj_length)):
        start_time = time.time()

        z, vel = estimator(obs.detach()[:, train_cfg.estimator.prop_start - (env_cfg.env.prop_hist_len-1) * train_cfg.estimator.prop_dim :train_cfg.estimator.prop_start + train_cfg.estimator.prop_dim])
        latent = torch.cat([z,vel],dim = 1)
        obs = torch.cat((obs[:, :train_cfg.estimator.priv_start], latent), dim = 1)


        # if args.use_jit:
        #     actions = policy(obs.detach())
        # else:
        #     actions = policy(obs.detach(), hist_encoding=True)

        # z, vel = estimator(obs.detach()[:, :train_cfg.estimator.prop_dim])
        # latent = torch.cat([z,vel],dim = 1)
        # obs = torch.cat((obs[:, :train_cfg.estimator.priv_start], latent), dim = 1)

        actions = policy(obs.detach(), hist_encoding=True)
        
        obs, _, rews, dones, infos = env.step(actions.detach())



        base_vel = torch.cat((env.base_lin_vel[:, :2], env.base_ang_vel[:, 2].unsqueeze(1)), dim=-1)  # 合并线速度和角速度

        goal_tracking_loss = torch.nn.MSELoss()(base_vel, env.commands[:, :3])
        goal_tracking_loss_sum += goal_tracking_loss
        # print('goal_tracking_loss', goal_tracking_loss)

        termination = env.reset_buf * ~env.time_out_buf


        state_data  = torch.cat((env.base_ang_vel, env.base_lin_vel, torch.stack((env.roll, env.pitch), dim = 1), env.dof_pos, env.dof_vel, env.commands[:, :3]), dim = -1)
        extrem_data = state_data[termination]

        # print("extrem_data", extrem_data.shape)
    
        # print('numpy', extrem_data.detach().cpu().numpy().shape)

        if extrem_data.shape[0] > 0:  # 对于普通 Python 列表或其他可迭代对象，检查长度是否大于 0
            all_extrem_data.append(extrem_data.detach().cpu().numpy())
            # print(all_extrem_data)
            # print(extrem_data)
        # all_extrem_data.append(extrem_data.detach().cpu().numpy() if isinstance(extrem_data, torch.Tensor) else extrem_data)

        # print('all_extrem_data', len(all_extrem_data))






        # # 将走出边界的 termination 置为 False
        # false_termination = termination & ~in_border

        # # 计算被 false 掉的 termination 的数量
        # out_border_num = torch.sum(false_termination).item()

        # out_border_sum += out_border_num
        # print("out_border", torch.sum(~in_border).item())

        # if torch.sum(~in_border).item() > 0:
        #     print(torch.sum(~in_border).item())
        #     break

        # print('termination_before', termination)
        termination[~in_border] = False
        # print('termination_after', termination)

        # 统计 True 的数量
        termination_num = torch.sum(termination).item()
        # if termination_num > 0:
        #     break

        # num_false = torch.sum(~termination).item()

        termination_sum += termination_num

        # print('termination_sum', termination_num)

        # success_rate =  num_false / (num_false + num_true)

        # print(termination)

        # print('amp_obs', env.extras["amp_obs"].shape)
        # amp_obs = env.extras["amp_obs"]

        # amp_demo_fetch_batch_size = 1000
        # amp_demo_obs = env.fetch_amp_obs_demo(amp_demo_fetch_batch_size)

        # human_like_loss = torch.nn.MSELoss()(amp_obs, amp_demo_obs)
        # human_like_loss_sum += human_like_loss

        # print('human_like_loss', human_like_loss)
        # print(demo_obs.shape)
        # in_border = ((-0.25 * env_cfg.terrain.terrain_length < env.root_states[:, 0]) & (env.root_states[:, 0] < total_length -0.25 * env_cfg.terrain.terrain_length)) & ((-0.25 * env_cfg.terrain.terrain_width < env.root_states[:, 1]) & (env.root_states[:, 1] < total_width-0.25 * env_cfg.terrain.terrain_width))
        in_border = ((0 < env.root_states[:, 0]) & (env.root_states[:, 0] < total_length)) & ((0 < env.root_states[:, 1]) & (env.root_states[:, 1] < total_width))
        # print("x", env.root_states[:, 0])
        # print("y", env.root_states[:, 1])

        Encoded_zmp = frequency_encoding(env.zmp_distance[env.lookat_id], 3)
        # print('zmp', env.zmp_distance[env.lookat_id])
        # print('Encoded_zmp', Encoded_zmp)

        if args.record_data:
            data_buf[env.lookat_id, i, 0] = env.commands[env.lookat_id, 0]    # command_x
            data_buf[env.lookat_id, i, 1] = env.commands[env.lookat_id, 1]    # command_y
            data_buf[env.lookat_id, i, 2] = env.commands[env.lookat_id, 2]    # command_yaw
            data_buf[env.lookat_id, i, 3] = env.base_lin_vel[env.lookat_id, 0]   # base x vel
            data_buf[env.lookat_id, i, 4] = env.base_lin_vel[env.lookat_id, 1]   # base y vel
            data_buf[env.lookat_id, i, 5] = env.base_lin_vel[env.lookat_id, 2]   # base z vel
            data_buf[env.lookat_id, i, 6] = env.base_ang_vel[env.lookat_id, 2]   # base yaw vel
            data_buf[env.lookat_id, i, 7:9] = env.contact_forces[env.lookat_id, env.feet_indices, 2]    # contact_force
            data_buf[env.lookat_id, i, 9] = env.zmp_distance[env.lookat_id]               # zmp feature

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
    
        # if (env.torque_limits[env.lookat_id] * 0.8 < env.torques[env.lookat_id]).any() or (-env.torque_limits[env.lookat_id] * 0.8 > env.torques[env.lookat_id]).any(): 

            # print("torque", env.torques[env.lookat_id])
        # time.sleep(env.dt/4)
        stop_time = time.time()

        # print(stop_time-start_time)

    print('Average_goal_tracking', goal_tracking_loss_sum / 1000.0)
    print('Success Rate', 1000 / (termination_sum + 1000))
    # print('Average_human_like_loss', human_like_loss_sum / 1000.0)
    # print(all_extrem_data)
    # print(np.array(all_extrem_data).shape)


    if args.record_data:
        file_name = "walk_data_0.6.npy"
        data = np.array(data_buf, dtype=object)
        np.save(file_name, data)
        print(f"Walk data has been saved as {file_name}")

    # if args.record_data:
    #     file_name = "extrem_data.npy"
    #     all_extrem_data = np.asanyarray(all_extrem_data, dtype=object)
    #     np.save(file_name, all_extrem_data)
    #     print(f"extrem_data has been saved as {file_name}")






if __name__ == '__main__':
    args = get_args()
    play(args)
