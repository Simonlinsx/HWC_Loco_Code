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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
class task():
    motion_task = 'walk'
    train_estimator = True

class G1MimicCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 1      #6144 8000
        episode_length_s = 50 # episode length in seconds

        force_type = 0

        # | ------------------------ |
        include_foot_contacts = True
        randomize_start_pos = False
        randomize_start_vel = False
        randomize_start_yaw = True
        rand_yaw_range = 0.2    # 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = True     # Can consider it!
        rand_pitch_range = 0.2    # 1.6 1.0

        if task.motion_task == 'walk':
            rand_vel = False

        else:
            rand_vel = True
            rand_lin_x_vel = [-1.0, 2.5]
            rand_lin_y_vel = [-1.0, 1.0]
            rand_lin_z_vel = [-0.4, 0.4]
            rand_ang_vel = [-1.0, 1.0]
            rand_pitch_range = 0.2
            randomize_start_pos = True
        


        # | ------------------------ |


        n_demo_steps = 2
        # n_demo = 9 + 3 + 3 + 3 +6*3  #observe height
        # n_demo = 9 + 3 + 3 + 3 +6*3
        n_demo = 9 + 3 + 3 + 3 + 6*3 + 6*3 +8
        n_demo = 9 + 3 + 3 + 3 + 6*3
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 0
        n_priv_latent = 4 + 1 + 23*2
        # n_proprio = 3 + 2 + 2 + 23*3 + 2# one hot

        n_depth = 0
        # // /// ////////////
        history_len = 11
        n_priv_latent = 0
        n_priv = 0

        if task.motion_task == 'walk':
            n_proprio = 3 + 2 + 2 + 23*3 +3 -2 + 3 + 2 - 2
            n_demo = 0
            episode_length_s = 20
        else:
            n_proprio = 3 + 2 + 2 + 23*3 +3 -2 + 3 + 2 - 2
            n_demo = 0
            episode_length_s = 12


        if task.train_estimator:
            n_priv = 3 + 1# default 3 vel     13: vel, diff
            n_latent = 16
        # ////// //////////

        # num_privileged_obs = n_proprio 
        prop_hist_len = 10    # 4    15
        n_feature = prop_hist_len * n_proprio

        n_decoder_out = n_latent + n_priv

        # num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_observations = n_feature + n_proprio + n_demo + n_priv

        # n_priv_proprio = n_proprio + 14 + 1 - 1
        n_priv_proprio = n_proprio
        num_privileged_obs = n_priv_proprio * prop_hist_len + n_priv_proprio + n_demo + n_priv + 51 + 132 + 5 + 23 * 2

        num_policy_actions = 23
        num_actions = 23


    
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
        vertical_scale = 0.005
        height = [0., 0.04]
        # height = [0., 0.]
        static_friction = 0.6   # 0.6
        dynamic_friction = 0.6
        downsampled_scale = 0.15
        y_range = [-0.1, 0.1]

        terrain_dict = {"flat": 0.2,
                        "rough": 0.2,
                        "discrete": 0.2,
                        "parkour_step": 0.,
                        "slop": 0.2,
                        "demo": 0.,
                        "down": 0.2,
                        "up": 0.2
                        }

        terrain_proportions = list(terrain_dict.values())
        measure_heights = True
        curriculum = True
        
        terrain_length = 20       # 20
        terrain_width = 12         #10 12 14
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 6 # number of terrain cols (types)


    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0.,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
        #    'torso_joint' : 0.,
           'waist_yaw_joint': 0.,
           'left_shoulder_pitch_joint' : 0.1, 
           'left_shoulder_roll_joint' : 0., 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.1,
           'right_shoulder_roll_joint' : -0.,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
           'left_wrist_roll_joint': 0.0,
           'right_wrist_roll_joint': 0.0
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                    'ankle': 40,
                    'torso': 200,
                    'shoulder': 30,
                    "elbow":30,
                    'waist_yaw_joint': 200,
                    'wrist_roll_joint': 30
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                    'torso': 5,
                    'shoulder': 1,
                    "elbow":1,
                    'waist_yaw_joint': 5,
                    'wrist_roll_joint': 1
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
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof_keybody.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_h2o.urdf'
        torso_name = "torso_link"
        foot_name = "ankle_roll"

        knee_name = "knee"    # 10.28
        elbow_name = "elbow"
        # penalize_contacts_on = ["shoulder", "elbow", "hip"]
        penalize_contacts_on = ["pelvis", "shoulder", "elbow", "hip"]
        # terminate_after_contacts_on = ["torso_link", ]#, "thigh", "calf"]
        # terminate_after_contacts_on = ['pelvis']
        # terminate_after_contacts_on = ["pelvis", "shoulder", "hip"]
        terminate_after_contacts_on = ["pelvis", "hip"]
        # terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            if task.motion_task == 'walk':
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
        # base_height_target = 0.98

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
        
        # 添加PD增益随机化的部分
        randomize_pd_gain = True  # 启用PD增益随机化
        kp_range = [0.75, 1.25]   # kp增益范围
        kd_range = [0.75, 1.25]   # kd增益范围
    
    class noise():
        add_noise = True
        noise_scale = 0.5 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.5     #0.3
            imu = 0.2
            gravity = 0.05
            # lin_vel = 0.05   # 原本没有的

    class task():
        motion_task = task.motion_task
        train_estimator = task.train_estimator

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]  11.1  TODO default 6.
        heading_command = True # True   if true: compute ang vel command from heading error

        lin_vel_clip = 0.01     # default 0.2
        ang_vel_clip = 0.01     # default 0.4

        class ranges:
            # lin_vel_x = [-0.6, 1.0] # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            # ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            # heading = [-1.6, 1.6]

            # Unitree gym rl
            lin_vel_x = [-0.3, 0.6] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-1.6, 1.6]

            # #Standing still
            # lin_vel_x = [0.0, 0.0] # min max [m/s]
            # lin_vel_y = [0.0, 0.0]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            # ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            # heading = [-1.6, 1.6]
    

class G1MimicCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False

        text_feat_input_dim = G1MimicCfg.env.n_feature

        text_feat_output_dim = 16
        feat_hist_len = G1MimicCfg.env.prop_hist_len
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]

        n_depth = G1MimicCfg.env.n_depth
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class estimator:
        train_with_estimated_states = True    #False
        learning_rate = 1.e-4
        # hidden_dims = [512, 256, 128]
        # hidden_dims = [128, 128]
        
        # encoder_hidden_dims = [512, 256]

        encoder_hidden_dims = [256, 128]
        # decoder_hidden_dims = [128, 64]
        decoder_hidden_dims = [256, 128, 64]
        priv_states_dim = G1MimicCfg.env.n_priv
        priv_start = G1MimicCfg.env.n_feature + G1MimicCfg.env.n_proprio + G1MimicCfg.env.n_demo + G1MimicCfg.env.n_scan
        
        prop_start = G1MimicCfg.env.n_feature
        prop_dim = G1MimicCfg.env.n_proprio

    class depth_encoder:
        if_depth = G1MimicCfg.depth.use_camera
        depth_shape = G1MimicCfg.depth.resized
        buffer_len = G1MimicCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = G1MimicCfg.depth.update_interval * 24
    

class G1MimicDistillCfgPPO( G1MimicCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]
        num_mini_batches = 4
