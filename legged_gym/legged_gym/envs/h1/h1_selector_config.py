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

import os

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

_MOTION_TASK_ENV_VAR = "LEGGED_GYM_MOTION_TASK"


def _get_motion_task_default():
    motion_task = os.environ.get(_MOTION_TASK_ENV_VAR, "walk")
    if motion_task not in ("walk", "recovery"):
        raise ValueError(f"Unsupported motion_task: {motion_task}")
    return motion_task


class task():
    motion_task = _get_motion_task_default()
    train_estimator = True
    use_gait = True

class H1SelectorCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 6144      #6144 8192
        episode_length_s = 50 # episode length in seconds

        # | ------------------------ |
        include_foot_contacts = True
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = True
        rand_yaw_range = 0.1    # 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = True     # Can consider it!
        rand_pitch_range = 0.2    # 1.6 1.0

        if task.motion_task == 'walk':
            rand_vel = False
            # 12 20
            randomize_start_pos = True
            extreme_flag = False
        else:
            rand_vel = True
            rand_lin_x_vel = [-1.5, 2.5]
            rand_lin_y_vel = [-1.5, 1.5]
            rand_lin_z_vel = [-0.4, 0.4]
            rand_ang_vel = [-1.0, 1.0]
            rand_pitch_range = 0.25
            randomize_start_pos = True
            
            extreme_flag = True

        # | ------------------------ |

        n_demo_steps = 2
        # n_demo = 9 + 3 + 3 + 3 +6*3  #observe height
        # n_demo = 9 + 3 + 3 + 3 +6*3
        n_demo = 9 + 3 + 3 + 3 + 6*3 + 6*3 +8
        n_demo = 9 + 3 + 3 + 3 + 6*3
        interval_demo_steps = 0.1

        n_scan = 0 #132
        n_priv = 0
        n_priv_latent = 4 + 1 + 19*2
        n_state_label = 0
        n_dr_label = 0
        # n_proprio = 3 + 2 + 2 + 19*3 + 2# one hot

        # /////////////////// #
        history_len = 6
        n_priv_latent = 0
        n_priv = 0

        if task.motion_task == 'walk':
            n_proprio = 3 + 2 + 2 + 19*3 +3 -2 + 3
            n_demo = 0
            episode_length_s = 14  # 20  18
        else:
            n_proprio = 3 + 2 + 2 + 19*3 +3 -2 + 3 
            n_demo = 0
            episode_length_s = 14
        
        if task.use_gait == True:
            n_proprio = n_proprio + 2

        if task.train_estimator:
            n_state_label = 3 + 1 + 1
            n_dr_label = 4 + 1 + 19 * 2 + 19 * 2 + 5
            n_priv = n_state_label + n_dr_label
            n_latent = 16
    
        # num_privileged_obs = n_proprio 
        prop_hist_len = 5
        n_feature = prop_hist_len * n_proprio

        n_decoder_out = n_latent + n_priv



        # num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        # num_observations = n_feature + n_proprio + n_priv
        num_observations = n_feature + n_proprio + n_priv

        n_priv_latent = n_dr_label
        # No hand
        n_priv_proprio = n_proprio + n_priv + n_priv_latent + 132 + 3 * 22 * 2
        n_priv_feature = n_priv_proprio * prop_hist_len
        # With hand
        # n_priv_proprio = n_proprio + n_priv + 43 + 132 + 5 + 19 * 2 + 3 * 20 * 2 
        # num_privileged_obs = n_priv_proprio * prop_hist_len + n_priv_proprio + n_priv + 43 + 132 + 5 + 19 * 2 + 3 * 22 * 2
        num_privileged_obs = n_priv_feature + n_priv_proprio

        num_policy_actions = 19


    
    class motion:
        if task.motion_task == 'walk':
            motion_curriculum = False
        else:
            motion_curriculum = True

        motion_type = "yaml"
        # motion_name = "motions_autogen_all_no_run_jump.yaml"
        motion_name = 'motions_autogen.yaml'

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = False

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]   0.005
        height = [0., 0.04]
        # height = [0., 0.]
        static_friction = 0.6
        dynamic_friction = 0.6
        downsampled_scale = 0.15
        y_range = [-0.1, 0.1]


        #  # Diverse
        # terrain_dict = {"flat": 0.2, 
        #                 "rough": 0.,
        #                 "discrete": 0.2,
        #                 "parkour step": 0.,
        #                 "slop": 0.2,
        #                 "demo": 0.2,
        #                 "down": 0.2,
        #                 "up": 0.2
        #                 }  

         # Diverse
        terrain_dict = {"flat": 0.2, 
                        "rough": 0.2,
                        "discrete": 0.2,
                        "parkour step": 0.,
                        "slop": 0.2,
                        "demo": 0.,
                        "down": 0.2,
                        "up": 0.2
                        }  

        terrain_proportions = list(terrain_dict.values())
        curriculum = False
        measure_heights = True
        
        terrain_length = 20       # 20
        terrain_width = 16         #10 12 
        num_rows= 20 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 12 # number of terrain cols (types)
    

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.1] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        lower_stand = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0.1, 
           'left_shoulder_roll_joint' : 0.1,   # 0.2
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 1.2,
           'right_shoulder_pitch_joint' : 0.1,
           'right_shoulder_roll_joint' : -0.1,  # -0.2
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 1.2,
        }

        # For demo test
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_joint' : -0.2,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_joint' : -0.2,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0.1, 
           'left_shoulder_roll_joint' : 0.1,   # 0.2
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 1.2,
           'right_shoulder_pitch_joint' : 0.1,
           'right_shoulder_roll_joint' : -0.1,  # -0.2
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 1.2,
        }



    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'hip_yaw': 200,
                    'hip_roll': 200,
                    'hip_pitch': 200,
                    'knee': 300,
                    'ankle': 40,
                    'torso': 200,
                    'shoulder': 30,
                    "elbow":30,
                    }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                    'hip_roll': 5,
                    'hip_pitch': 5,
                    'knee': 6,
                    'ankle': 2,
                    'torso': 5,
                    'shoulder': 1,
                    "elbow":1,
                    }  # [N*m/rad]  # [N*m*s/rad]
     
        action_filt = False
        action_cutfreq = 5.0    # 4.0

        action_scale = 0.25
        decimation = 10    # 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10
        # clip_actions = 100.
        # clip_observations = 100.

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_blue_red_custom_collision.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_h2o.urdf'
        torso_name = "torso_link"
        foot_name = "ankle"
        knee_name = "knee"    # 10.28
        elbow_name = "elbow"
        waist_name = "torso_joint"
        hip_joints = [
            "left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint",
            "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint",
        ]
        knee_joints = ["left_knee_joint", "right_knee_joint"]
        ankle_joints = ["left_ankle_joint", "right_ankle_joint"]
        shoulder_joints = [
            "left_shoulder_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint",
            "right_shoulder_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint",
        ]
        elbow_joints = ["left_elbow_joint", "right_elbow_joint"]
        waist_joints = ["torso_joint"]
        # penalize_contacts_on = ["shoulder", "elbow", "hip"]
        penalize_contacts_on = ["pelvis", "shoulder", "elbow", "hip", "knee"]
        # terminate_after_contacts_on = ["torso_link", ]#, "thigh", "calf"]
        # terminate_after_contacts_on = ['pelvis']
        terminate_after_contacts_on = ["pelvis", "hip"]
        # terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False

  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            if task.motion_task == 'walk':
                
                # Gait reward
                # joint_pos = 0.8      # 1.6    0.8   
                # feet_clearance = 0.4     # 1.   0.5    0.4
                # feet_contact_number = 0.5     # 1.2    0.6    0.5
                # feet_air_time = 10.0
                # feet_stumble = -2
                # feet_drag = -0.1

                # Command tracking reward
                tracking_lin_vel = 1.2    # 1.0 10
                tracking_ang_vel = 1.0    # 0.5 5
                # vel_mismatch_exp = 0.5  # lin_z; ang x,y
                # low_speed = 0.2
                # track_vel_hard = 0.5

                # Regularization reward
                # feet_distance = 0.2
                # knee_distance = 0.2
                # default_joint_pos = 0.2    # 0.5   0.2
                # upper_joint_pos = 2.0       # 2.0  10.0   4.0    0.1
                # orientation = 0.4        # 1.0  0.4   0.2   0.5


                # Safe reward
                # feet_contact_forces = -0.01
                # action_smoothness = -0.002
                # torques = -1e-5
                # dof_vel = -5e-4
                # dof_acc = -1e-7
                # base_acc = 0.2
                # collision = -10.0
                alive = 10 # 6     

                # Recovery reward
                # zmp_distance = 0.5             
                termination = -100    # -200  -100  -20   


            else:
                # For tracking target
                alive = 1
                tracking_lin_vel = 6
                tracking_demo_yaw = 1
                tracking_demo_roll_pitch = 1
                orientation = -2
                tracking_demo_dof_pos = 3
                tracking_demo_key_body = 2
                lin_vel_z = -1.0
                ang_vel_xy = -0.4
                dof_acc = -3e-7
                # collision = -10.
                action_rate = -0.1
                energy = -1e-3
                dof_error = -0.1
                feet_stumble = -2
                feet_drag = -0.1
                dof_pos_limits = -10.0
                feet_air_time = 10
                # feet_height = 0           # 2
                feet_force = -3e-3

        # humanoid gym
        # //
        # max_contact_force = 1000
        base_height_target = 0.90    # 0.90
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06       # m
        cycle_time = 0.64                # sec
        # //

        tracking_sigma = 5.0
        only_positive_rewards = True     # False
        clip_rewards = False        # True
        soft_dof_pos_limit = 0.98
        stability_margin_threshold = 0.0
        use_zmp_cost = task.motion_task == 'recovery'
        zmp_cost_type = "margin"
        zmp_no_contact_cost = 0.0
        support_polygon_front = 0.14
        support_polygon_back = 0.08
        support_polygon_left = 0.015
        support_polygon_right = 0.015
        support_polygon_shrink = 0.003
        # base_height_target = 0.98

        Gait_reward = [
            "joint_pos",           # 用于控制关节位置的奖励
            "feet_clearance",      # 确保步态中脚的清晰抬起
            "feet_contact_number" # 奖励正确的脚部接触数量
        ]

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.90
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
        
        added_com_range_x = [-0.10, 0.10]   # [-0.15, 0.15]
        added_com_range_y = [-0.10, 0.10]
        added_com_range_z = [-0.10, 0.10]
        

        # 添加PD增益随机化的部分
        randomize_pd_gain = True  # 启用PD增益随机化
        kp_range = [0.9, 1.1]   # kp增益范围
        kd_range = [0.9, 1.1]   # kd增益范围

        # 可能就是这个影响了算法性能
        delay_update_global_steps = 24 * 35000   # 24 * 8000   35000
        action_curr_step = [1, 2, 3, 4]

        velocity_sample_global_steps = 24 * 40000
        # For deployment
        # delay_update_global_steps = 24 * 2000


        push_robots = True
        push_interval_s = 6     # 6
        max_push_vel_xy = 3.0   # 0.5   1.0  2.0
        max_push_ang_vel = 1.0   # 0.3
        dynamic_randomization = 0.02
        randomize_pd_gain = True
        max_force = 20
        max_torque = 20


    class noise():
        add_noise = True
        noise_scale = 1.0 # scales other values    2.0 for recovery policy
        class noise_scales():
            # dof_pos = 0.01   
            # dof_vel = 0.1
            # ang_vel = 0.3     #0.3
            # imu = 0.1
            # gravity = 0.05    # 0.05
            dof_pos = 0.02
            dof_vel = 0.20
            ang_vel = 0.5     #0.3
            imu = 0.2
            gravity = 0.1
            lin_vel = 0.1   # 原本没有的


    class task():
        motion_task = task.motion_task
        train_estimator = task.train_estimator
        use_gait = task.use_gait

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]  11.1  TODO default 6.
        heading_command = False # True   if true: compute ang vel command from heading error

        lin_vel_clip = 0.1     # default 0.2
        ang_vel_clip = 0.2     # default 0.4

        class ranges:
            lin_vel_x = [-0.6, 2.5] # min max [m/s]   2.0
            lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-1.6, 1.6]

class H1SelectorCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "Selector_Trainer"
        selector_batch_size = 512
        selector_updates_per_iteration = 32
        selector_target_update_frequency = 100
        selector_replay_capacity = 2000000
        selector_max_grad_norm = 1.0
        selector_switch_penalty = 0.00002
        selector_epsilon_start = 0.1
        selector_epsilon_min = 0.0002
        selector_epsilon_decay = 0.998

    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False

        text_feat_input_dim = H1SelectorCfg.env.n_feature

        text_feat_output_dim = 16
        feat_hist_len = H1SelectorCfg.env.prop_hist_len

        selector_input = H1SelectorCfg.env.n_feature + H1SelectorCfg.env.n_proprio + H1SelectorCfg.env.n_decoder_out + 1
        selector_dropout_prob = 0.0
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01   # try 0.01 or 0.001 ?

        # # humanoid gym
        # entropy_coef = 0.001
        # learning_rate = 1e-5
        # num_learning_epochs = 2
        # gamma = 0.994
        # lam = 0.9
        # num_mini_batches = 4

    class estimator:
        train_with_estimated_states = True    #False
        learning_rate = 1.e-4
        hidden_dims = [512, 256, 128]
        # hidden_dims = [128, 128]


        history_len = H1SelectorCfg.env.prop_hist_len
        priv_states_dim = H1SelectorCfg.env.n_priv
        priv_start = H1SelectorCfg.env.n_feature + H1SelectorCfg.env.n_proprio
        
        prop_start = H1SelectorCfg.env.n_feature
        prop_dim = H1SelectorCfg.env.n_proprio
