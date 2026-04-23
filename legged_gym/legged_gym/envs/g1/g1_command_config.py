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
    motion_task = _get_motion_task_default()    # walk or recovery
    train_estimator = True
    use_depth = False
    use_gait = False

class G1CommandCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096      #6144 8000 4096
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
            extreme_flag = False

        else:
            rand_vel = True
            rand_lin_x_vel = [-1.0, 2.5]
            rand_lin_y_vel = [-1.0, 1.0]
            rand_lin_z_vel = [-0.4, 0.4]
            rand_ang_vel = [-1.0, 1.0]
            rand_pitch_range = 0.2
            randomize_start_pos = True
            extreme_flag = True

        history_len = 6

        n_proprio = 3 + 2 + 23 * 3 + 3 + 3
        n_demo = 0
        episode_length_s = 14

        if task.train_estimator:
            n_priv = 3 + 1 + 8 # default 3 vel     13: vel, diff
            n_latent = 16

        prop_hist_len = 5    # 4    15
        n_feature = prop_hist_len * n_proprio

        n_decoder_out = n_latent + n_priv

        num_observations = n_feature + n_proprio + n_priv
        n_priv_latent = 51 + 23 * 2 + 5

        n_priv_proprio = n_proprio + n_priv + n_priv_latent + 132 

        # motion_features: historical proprioception
        n_priv_feature = n_priv_proprio * prop_hist_len
        num_privileged_obs = n_priv_feature + n_priv_proprio

        num_policy_actions = 23 
        num_actions = 23  


    # For mimic or amp
    class motion:
        motion_curriculum = False
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
        vertical_scale = 0.005
        height = [0., 0.04]
        static_friction = 1.0   # 0.6
        dynamic_friction = 1.0
        downsampled_scale = 0.15
        y_range = [-0.1, 0.1]

        terrain_dict = {"flat": 0.8,
                        "rough": 0.2,
                        "discrete": 0.,
                        "parkour_step": 0.,
                        "slop": 0.,
                        "demo": 0.,
                        "down": 0.,
                        "up": 0.
                        }

        terrain_proportions = list(terrain_dict.values())
        measure_heights = True
        curriculum = False
        
        terrain_length = 20       # 20
        terrain_width = 16         #10 12 14
        num_rows= 20 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 12 # number of terrain cols (types)


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint' : -0.1,   # 0
            'left_hip_roll_joint' : 0, 
            'left_hip_yaw_joint' : 0., 
            'left_knee_joint' : 0.3,     
            'left_ankle_pitch_joint' : -0.2, 
            'left_ankle_roll_joint' : 0, 
            'right_hip_pitch_joint' : -0.1, 
            'right_hip_roll_joint' : 0, 
            'right_hip_yaw_joint' : 0., 
            'right_knee_joint' : 0.3, 
            'right_ankle_pitch_joint': -0.2, 
            'right_ankle_roll_joint' : 0, 
            'waist_yaw_joint' : 0.,         # 12
            'waist_roll_joint' : 0., 
            'waist_pitch_joint' : 0., 
            'left_shoulder_pitch_joint': 0.1,
            'left_shoulder_roll_joint': 0.2, #3.14159/180*60,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.3, #3.14159/2,
            'right_shoulder_pitch_joint': 0.1, 
            'right_shoulder_roll_joint': -0.2, #-3.14159/180*60,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.3 #3.14159/2,
        }


    class sim( LeggedRobotCfg.sim ):
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
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'      
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 20,
                     "waist": 300,
                     "shoulder_pitch": 90,
                     "shoulder_roll": 60,
                     "shoulder_yaw": 20,
                     "elbow": 100
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 0.2,
                     "waist": 5,
                     'shoulder_pitch': 2.0,
                     'shoulder_roll': 1.0,
                     'shoulder_yaw': 0.4,
                     "elbow": 1
                     }  # [N*m/rad]  # [N*m*s/rad]

        action_filt = False
        action_cutfreq = 5.0    # 4.0

        action_scale = 0.25
        decimation = 4 

    class normalization( LeggedRobotCfg.normalization):
        clip_actions = 10

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0_anneal_23.urdf'
        # link names
        torso_name = "torso_link"
        foot_name = "ankle_roll"
        knee_name = "knee"
        elbow_name = "elbow"

        # joint names
        hip_joints = ["left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint"]
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint"]
        shoulder_joints = ['left_shoulder_roll_joint', 'left_shoulder_pitch_joint', 'left_shoulder_yaw_joint',
                          'right_shoulder_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_yaw_joint']
        elbow_joints = ['left_elbow_joint', 'right_elbow_joint']
        waist_joints = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']

        # penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            tracking_lin_vel = 3.0   # 1.0 3.0
            tracking_ang_vel = 4.0  # 0.5 4.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -2.0 # -1.0 
            base_height = -10.0 # -10.0
            dof_acc = -5e-8   # -5e-8  
            dof_vel = -4e-5   # -4e-5  
            feet_air_time = 10.0   # 0.0
            collision = - 1.0
            action_rate = -0.01    # -0.01
            dof_pos_limits = -5.0
            alive = 0.40   # 0.15

            contact_no_vel = -0.2
            feet_swing_height = -20.0  
            contact = 0.18 

            feet_distance = 0.4 
            knee_distance = 0.4 

            # pos regular
            # upper_pos = -0.2
            hip_pos = -1.0
            shoulder_pos = -0.5  # -1.0 
            elbow_pos = -0.5 # -1.0
            waist_pos = -2.0  # -1.0
            ankle_pos = -1.0 # -0.5 
            # knee_pos = -0.5  

            feet_stumble = -1.5
            feet_slip = -0.25
            # feet_parallel = -5.0

            # stand_still = -1.0

        max_contact_force = 400.
        base_height_target = 0.78    # 0.90
        min_dist = 0.2
        max_dist = 0.35

        # put some settings here for LLM parameter tuning (not used)
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06       # m
        cycle_time = 0.64                # sec 

        tracking_sigma = 4.0
        only_positive_rewards = False     # False
        clip_rewards = False        # True
        soft_dof_pos_limit = 0.98

        Gait_reward = [
            "joint_pos",           # 用于控制关节位置的奖励
            "feet_clearance",      # 确保步态中脚的清晰抬起
            "feet_contact_number" # 奖励正确的脚部接触数量
        ]

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.95
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_gravity = True
        gravity_rand_interval_s = 10
        gravity_range = [-0.1, 0.1]
        
        # 添加PD增益随机化的部分
        randomize_pd_gain = True  # 启用PD增益随机化
        kp_range = [0.80, 1.20]   # kp增益范围
        kd_range = [0.80, 1.20]   # kd增益范围

        randomize_base_com = True
        added_com_range_x = [-0.1, 0.1]   # [-0.15, 0.15]
        added_com_range_y = [-0.1, 0.1]
        added_com_range_z = [-0.1, 0.1]

        velocity_sample_global_steps = 24 * 40000

        if task.motion_task == 'walk':
            push_robots = True
            push_interval_s = 8
            max_push_vel_xy = 1.0   # 0.5   1.0  2.0
            max_push_ang_vel = 0.5   # 0.3
            dynamic_randomization = 0.02
        else:
            push_robots = True
            push_interval_s = 6
            max_push_vel_xy = 2.0   # 0.5   1.0  2.0
            max_push_ang_vel = 0.8   # 0.3
            dynamic_randomization = 0.02


        randomize_friction = True
        friction_range = [0.6, 2.0]  # [0.6, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1., 5]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.3

        randomize_motor = True
        motor_strength_range = [0.9, 1.1] # [0.8, 1.2]

        delay_update_global_steps = 24 * 80000
        action_delay = False
        action_curr_step = [2, 3, 4, 5]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8
        randomize_link_mass = True
        link_mass_range = [0.9, 1.1] # *factor
        randomize_link_body_names = [
            'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 
            'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',  'torso_link',
        ]

    class noise():
        add_noise = True
        noise_scale = 1.0 # scales other values
        class noise_scales():
            dof_pos = 0.02
            dof_vel = 2.0
            ang_vel = 0.5
            imu = 0.2
            gravity = 0.05

    class task():
        motion_task = task.motion_task
        train_estimator = task.train_estimator


    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]  11.1  TODO default 6.
        if task.motion_task == 'recovery':
            resampling_time = 100.
        heading_command = False # True   if true: compute ang vel command from heading error

        lin_vel_clip = 0.1     # default 0.2
        ang_vel_clip = 0.1     # default 0.4

        class ranges:
            lin_vel_x = [-1.0, 1.5] # min max [m/s]   2.0
            lin_vel_y = [-1.0, 1.0]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-1.6, 1.6]



class G1CommandCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = True

        text_feat_input_dim = G1CommandCfg.env.n_feature

        text_feat_output_dim = 32 # 16
        feat_hist_len = G1CommandCfg.env.prop_hist_len
    

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01   # try 0.01 or 0.001 ?
        schedule = 'fixed' # could be adaptive, fixed

    class estimator:
        train_with_estimated_states = True    #False
        learning_rate = 1.e-4

        # For Deployment
        encoder_hidden_dims = [256, 128, 64]
        decoder_hidden_dims = [256, 128, 64]

        # # For simulation
        # encoder_hidden_dims = [512, 256]
        # decoder_hidden_dims = [256, 128, 64]

        n_demo = G1CommandCfg.env.n_demo
        priv_latent_dim = G1CommandCfg.env.n_priv_latent

        history_len = G1CommandCfg.env.prop_hist_len
        priv_states_dim = G1CommandCfg.env.n_priv
        priv_start = G1CommandCfg.env.n_feature + G1CommandCfg.env.n_proprio
        
        prop_start = G1CommandCfg.env.n_feature
        prop_dim = G1CommandCfg.env.n_proprio
        priv_prop_start = G1CommandCfg.env.n_priv_feature
