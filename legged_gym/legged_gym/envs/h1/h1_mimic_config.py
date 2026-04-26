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
    motion_task = 'recovery'  # motion   recovery
    train_estimator = True


class H1MimicCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096      #6144 8000
        episode_length_s = 50 # episode length in seconds

        n_demo_steps = 2
        # Only upper
        # n_demo = 9 + 3 + 3 + 3 + 6*3
        # Whole-body tracking
        n_demo = 17 + 3 + 3 + 3 + 12*3

        interval_demo_steps = 0.1

        n_scan = 0#132
        n_priv = 0
        n_priv_latent = 4 + 1 + 19*2
        # n_proprio = 3 + 2 + 2 + 19*3 + 2# one hot


        # // /// ////////////
        history_len = 11
        n_priv_latent = 0
        n_priv = 0

        n_proprio = 3 + 2 + 2 + 19*3  

        extreme_flag = False

        if task.train_estimator:
            n_priv = 3 + 8 + 1 # default 3 vel     13: vel, diff
            n_latent = 16
        # ///////////       /////////////       //////////

        n_decoder_out = n_latent + n_priv

        prop_hist_len = 10    # 4
        n_feature = prop_hist_len * n_proprio

        num_observations = n_feature + n_proprio + n_demo + n_priv

        n_priv_latent = 43 + 19 * 2 + 5

        n_priv_proprio = n_proprio + n_demo + n_priv_latent + 132 + n_priv
        n_priv_feature = n_priv_proprio * prop_hist_len
        num_privileged_obs = n_priv_feature + n_priv_proprio
        
        num_policy_actions = 19
    

    class motion:

        motion_curriculum = True

        motion_type = "yaml"
        # motion_name = "motions_autogen_all_no_run_jump.yaml"
        motion_name = 'motions_autogen.yaml'

        global_keybody = False
        global_keybody_reset_time = 2

        num_envs_as_motions = False

        no_keybody = False
        regen_pkl = True

        step_inplace_prob = 0.05
        resample_step_inplace_interval_s = 10

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.15 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.04, 0.12]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.
        terrain_width = 20
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 10 # number of terrain cols (types)

        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]   0.005
        height = [0., 0.04]
        # height = [0., 0.]
        static_friction = 0.6
        dynamic_friction = 0.6
        downsampled_scale = 0.15
        y_range = [-0.1, 0.1]


         # Diverse
        terrain_dict = {"flat": 0.8, 
                        "rough": 0.2,
                        "discrete": 0.,
                        "parkour step": 0.,
                        "slop": 0.,
                        "demo": 0.,
                        "down": 0.,
                        "up": 0.
                        }  

        terrain_proportions = list(terrain_dict.values())
        curriculum = False
        measure_heights = True
        
        terrain_length = 20       # 20
        terrain_width = 16         #10 12 
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 8 # number of terrain cols (types)


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.1] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]


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
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 200,
                    'hip_roll': 200,
                    'hip_pitch': 200,
                    'knee': 200,
                    'ankle': 80,
                    'torso': 200,
                    'shoulder': 30,
                    "elbow":30,
                    }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                    'hip_roll': 5,
                    'hip_pitch': 5,
                    'knee': 5,
                    'ankle': 2,
                    'torso': 5,
                    'shoulder': 1,
                    "elbow":1,
                    }  # [N*m/rad]  # [N*m*s/rad]

        action_filt = False
        action_cutfreq = 5.0    # 4.0

        action_scale = 0.25
        decimation = 4    # 4

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10
        # clip_observations = 18.
        # clip_actions = 18.
        
    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_blue_red_custom_collision.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision_exbody.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_h2o.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_linker/h1_linker_fixed.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_linker/h1_linker_with_keypoints.urdf'
        torso_name = "torso_link"
        foot_name = "ankle"
        knee_name = "knee"    # 10.28
        # penalize_contacts_on = ["shoulder", "elbow", "hip"]
        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ["torso_link", ]#, "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.90
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            if task.motion_task == 'recovery':
                # For tracking target   50 HZ
                alive = 2.0   # 1  1.5
                tracking_lin_vel = 6.0
                tracking_demo_yaw = 1.0
                tracking_demo_roll_pitch = 1.0
                orientation = -2
                tracking_demo_dof_pos = 2.4   # 3
                tracking_demo_key_body = 1.6  # 2
                lin_vel_z = -1.0
                ang_vel_xy = -0.4
                dof_acc = -3e-7
                collision = -10.
                action_rate = -0.1
                energy = -1e-3
                dof_error = -0.1
                feet_stumble = -2
                feet_drag = -0.1
                dof_pos_limits = -10.0
                feet_air_time = 10
                # feet_height = 0           # 2
                feet_force = -3e-3
                zmp_distance = 2.0
                termination = -100
                lower_stand = -0.1 


            else:
                # For tracking target   50 HZ
                alive = 1
                tracking_lin_vel = 6
                tracking_demo_yaw = 1
                tracking_demo_roll_pitch = 1
                orientation = -2
                tracking_demo_dof_pos = 3     # 3 
                tracking_demo_key_body = 2.5   # 2  2.5
                lin_vel_z = -1.0
                ang_vel_xy = -0.4
                dof_acc = -3e-7
                collision = -10.
                action_rate = -0.1
                energy = -1e-3
                dof_error = -0.1
                feet_stumble = -2
                feet_drag = -0.1
                dof_pos_limits = -10.0
                feet_air_time = 10
                # feet_height = 0           # 2
                feet_force = -3e-3


                # zmp_distance = 2.0

                # # # For tracking target    100 HZ
                # alive = 1
                # tracking_lin_vel = 6
                # tracking_demo_yaw = 1
                # tracking_demo_roll_pitch = 1
                # orientation = -2.0
                # tracking_demo_dof_pos = 3
                # tracking_demo_key_body = 3  # 2
                # lin_vel_z = -0.2
                # ang_vel_xy = -0.05
                # dof_acc = -6e-8
                # # collision = -10.
                # action_rate = -0.05
                # energy = -1e-3
                # dof_error = -0.1
                # feet_stumble = -2
                # feet_drag = -0.1
                # dof_pos_limits = -10.0
                # feet_air_time = 2
                # # feet_height = 0           # 2
                # feet_force = -3e-3

        # humanoid gym
        # 
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
        only_positive_rewards = False
        clip_rewards = True
        soft_dof_pos_limit = 0.98
        # base_height_target = 0.98
    

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]

        delay_update_global_steps = 24 * 40000

        added_mass_range = [-1., 10]   # [-1., 8]
        
        # added_com_range_x = [-0.10, 0.10]   # [-0.15, 0.15]
        # added_com_range_y = [-0.10, 0.10]
        # added_com_range_z = [-0.10, 0.10]

        # added_com_range_x = [-0.15, 0.15]   # [-0.15, 0.15]
        # added_com_range_y = [-0.10, 0.10]
        # added_com_range_z = [-0.15, 0.15]

        added_com_range = [-0.15, 0.15]    # [-0.10, 0.10]

        # 添加PD增益随机化的部分
        randomize_pd_gain = True  # 启用PD增益随机化
        # kp_range = [0.9, 1.1]   # kp增益范围
        # kd_range = [0.9, 1.1]   # kd增益范围

        kp_range = [0.8, 1.2]   # kp增益范围
        kd_range = [0.8, 1.2]   # kd增益范围

        if task.motion_task == 'motion':
            push_robots = True
            push_interval_s = 8
            max_push_vel_xy = 2.0   # 0.5   1.0  2.0
            max_push_ang_vel = 0.8   # 0.3
            dynamic_randomization = 0.02
            max_force = 10
            max_torque = 10

        else:
            push_robots = True
            push_interval_s = 6     # 6
            max_push_vel_xy = 3.0   # 0.5   1.0  2.0
            max_push_ang_vel = 1.0   # 0.3
            dynamic_randomization = 0.02
            max_force = 20
            max_torque = 20

    class noise():
        add_noise = True
        noise_scale = 1.0 # scales other values
        class noise_scales():
            dof_pos = 0.01
            dof_vel = 0.15
            ang_vel = 0.5     #0.3
            imu = 0.2
            gravity = 0.05
            lin_vel = 0.05   # 原本没有的

        # class noise_scales():
        #     dof_pos = 0.02
        #     dof_vel = 0.20
        #     ang_vel = 0.5     #0.3
        #     imu = 0.25       # 0.2
        #     gravity = 0.2   # 0.1
        #     lin_vel = 0.1   # 原本没有的


    class task():
        motion_task = task.motion_task
        train_estimator = task.train_estimator

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before commands are resampled [s]
        heading_command = True # True   if true: compute ang vel command from heading error

        lin_vel_clip = 0.1     # default 0.2
        ang_vel_clip = 0.1     # default 0.4

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

    class sim:
        dt =  0.005    # default 0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 1.0 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)



class H1MimicCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'

        max_iterations = 80000
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = False

        text_feat_input_dim = H1MimicCfg.env.n_feature

        text_feat_output_dim = 16
        feat_hist_len = H1MimicCfg.env.prop_hist_len
        # actor_hidden_dims = [1024, 512]
        # critic_hidden_dims = [1024, 512]
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01   # 0.005

    class estimator:
        train_with_estimated_states = True    #False
        learning_rate = 1.e-4
        hidden_dims = [128, 64]

        encoder_hidden_dims = [512, 256]
        decoder_hidden_dims = [256, 128, 64]


        n_demo = H1MimicCfg.env.n_demo
        history_len = H1MimicCfg.env.prop_hist_len
        priv_latent_dim = H1MimicCfg.env.n_priv_latent
        priv_states_dim = H1MimicCfg.env.n_priv
        priv_start = H1MimicCfg.env.n_feature + H1MimicCfg.env.n_proprio + H1MimicCfg.env.n_demo + H1MimicCfg.env.n_scan
        
        prop_start = H1MimicCfg.env.n_feature
        prop_dim = H1MimicCfg.env.n_proprio
        priv_prop_start = H1MimicCfg.env.n_priv_feature



class H1MimicDistillCfgPPO( H1MimicCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]
        num_mini_batches = 4
