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
    use_depth = False
    use_gait = True

class G1SelectorCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096      #6144 8000
        episode_length_s = 50 # episode length in seconds


        if task.motion_task == 'walk':
            rand_vel = False
            randomize_start_pos = True
            extreme_flag = False
        else:
            rand_vel = True
            rand_lin_x_vel = [-1.0, 2.5]
            rand_lin_y_vel = [-1.0, 1.0]
            rand_lin_z_vel = [-0.2, 0.2]
            rand_ang_vel = [-1.0, 1.0]
            rand_pitch_range = 0.2
            randomize_start_pos = True
            extreme_flag = True
        

        n_demo_steps = 2
        # n_demo = 9 + 3 + 3 + 3 +6*3  #observe height
        # n_demo = 9 + 3 + 3 + 3 +6*3
        n_demo = 9 + 3 + 3 + 3 + 6*3 + 6*3 +8
        n_demo = 9 + 3 + 3 + 3 + 6*3
        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 0
        n_priv_latent = 4 + 1 + 19*2
        # n_proprio = 3 + 2 + 2 + 19*3 + 2# one hot

        # // /// ////////////
        history_len = 21
        n_priv_latent = 0
        n_priv = 0

        if task.motion_task == 'walk':
            n_proprio = 3 + 2 + 2 + 19*3 +3 -2 + 3 + 2
            n_demo = 0
            episode_length_s = 14      # 这个影响了效果？
        else:
            n_proprio = 3 + 2 + 2 + 19*3  

        if task.use_depth:
            n_depth = 58 * 87
        else:
            n_depth = 0
            
        if task.train_estimator:
            n_priv = 3 + 1 + 43 # default 3 vel     13: vel, diff
            n_latent = 16
        # ////// //////////

        # num_privileged_obs = n_proprio 
        prop_hist_len = 20    # 4
        n_feature = prop_hist_len * n_proprio

        n_decoder_out = n_latent + n_priv

        # num_observations = n_feature + n_proprio + n_demo + n_scan + history_len*n_proprio + n_priv_latent + n_priv
        num_observations = n_feature + n_proprio + n_priv

        n_priv_proprio = n_proprio + n_priv + 43 + 132 + 5 + 19 * 2 + 3 * 22 * 2
        # num_privileged_obs = n_priv_proprio * prop_hist_len + n_priv_proprio + n_priv + 43 + 132 + 5 + 19 * 2 + 3 * 22 * 2
        num_privileged_obs = n_priv_proprio * prop_hist_len + n_priv_proprio

        num_policy_actions = 19

    
    class motion:
        if task.motion_task == 'walk':
            motion_curriculum = False
        else:
            motion_curriculum = True

        motion_type = "yaml"
        # motion_name = "motions_autogen_all_no_run_jump.yaml"
        motion_name = 'motions_autogen.yaml'

        global_keybody = True
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
                        "down": 0.,
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

        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #    'left_hip_yaw_joint' : 0. ,   
        #    'left_hip_roll_joint' : 0,               
        #    'left_hip_pitch_joint' : -0.4,         
        #    'left_knee_joint' : 0.8,       
        #    'left_ankle_joint' : -0.4,     
        #    'right_hip_yaw_joint' : 0., 
        #    'right_hip_roll_joint' : 0, 
        #    'right_hip_pitch_joint' : -0.4,                                       
        #    'right_knee_joint' : 0.8,                                             
        #    'right_ankle_joint' : -0.4,                                     
        #    'torso_joint' : 0., 
        #    'left_shoulder_pitch_joint' : 0., 
        #    'left_shoulder_roll_joint' : 0., 
        #    'left_shoulder_yaw_joint' : 0.,
        #    'left_elbow_joint'  : 0.8,
        #    'right_shoulder_pitch_joint' : 0.,
        #    'right_shoulder_roll_joint' : -0.,
        #    'right_shoulder_yaw_joint' : 0.,
        #    'right_elbow_joint' : 0.8,
        # }
        # For demo test
        default_joint_angles = { # = target angles [rad] when action = 0.0
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
           'left_shoulder_roll_joint' : 0.1, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 1.2,
           'right_shoulder_pitch_joint' : 0.1,
           'right_shoulder_roll_joint' : -0.1,
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
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/G1_blue_red_custom_collision.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/G1_custom_collision.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/G1.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/G1/G1_h2o.urdf'
        torso_name = "torso_link"
        foot_name = "ankle"
        knee_name = "knee"    # 10.28
        elbow_name = "elbow"
        waist_name = "waist"
        hip_joints = [
            "left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint",
            "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint",
        ]
        knee_joints = ["left_knee_joint", "right_knee_joint"]
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        shoulder_joints = [
            "left_shoulder_roll_joint", "left_shoulder_pitch_joint", "left_shoulder_yaw_joint",
            "right_shoulder_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_yaw_joint",
        ]
        elbow_joints = ["left_elbow_joint", "right_elbow_joint"]
        waist_joints = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
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
                tracking_lin_vel = 1.0    # 1.0 10
                tracking_ang_vel = 0.5    # 0.5 5
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
                # alive = 8 # 6     

                # Recovery reward
                # zmp_distance = 0.2             
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


        added_com_range_x = [-0.1, 0.1]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.20, 0.20]
        
        # 添加PD增益随机化的部分
        randomize_pd_gain = False  # 启用PD增益随机化
        kp_range = [0.8, 1.2]   # kp增益范围
        kd_range = [0.8, 1.2]   # kd增益范围

        delay_update_global_steps = 24 * 10000   # 24 * 8000

        push_robots = True
        push_interval_s = 8    # 6 8
        max_push_vel_xy = 1.5   # 0.5   1.0  2.0  0.8
        max_push_ang_vel = 0.5   # 0.3   0.5
        dynamic_randomization = 0.02

    class noise():
        add_noise = True
        noise_scale = 1.0 # scales other values
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
        resampling_time = 10. # time before commands are resampled [s]
        heading_command = False # True   if true: compute ang vel command from heading error

        lin_vel_clip = 0.1     # default 0.2
        ang_vel_clip = 0.2     # default 0.4

        class ranges:
            lin_vel_x = [-0.6, 2.5] # min max [m/s]   2.0
            lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-1.6, 1.6]

                
    class depth:
        use_camera = task.use_depth
        # camera_num_envs = 192
        # camera_terrain_num_rows = 10
        # camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [0, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
    
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

class G1SelectorCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "Selector_Trainer"

    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False

        text_feat_input_dim = G1SelectorCfg.env.n_feature

        text_feat_output_dim = 16
        feat_hist_len = G1SelectorCfg.env.prop_hist_len

        selector_input = G1SelectorCfg.env.n_feature + G1SelectorCfg.env.n_proprio + G1SelectorCfg.env.n_decoder_out
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
        n_depth = G1SelectorCfg.env.n_depth
    
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

        priv_states_dim = G1SelectorCfg.env.n_priv
        priv_start = G1SelectorCfg.env.n_feature + G1SelectorCfg.env.n_proprio
        
        prop_start = G1SelectorCfg.env.n_feature
        prop_dim = G1SelectorCfg.env.n_proprio



class G1MimicDistillCfgPPO( G1SelectorCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]
        num_mini_batches = 4
