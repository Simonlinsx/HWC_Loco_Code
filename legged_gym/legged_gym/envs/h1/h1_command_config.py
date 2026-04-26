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
    # 'walk' refers to goal-tracking policy
    # 'recovery' refers to safety recovery policy
    use_gait = True

class H1CommandCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096      #6144 8192 
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
            # Warm-start recovery from milder initial states before re-introducing
            # aggressive reset distributions.
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
        n_demo = 9 + 3 + 3 + 3 + 6*3 + 6*3 +8
        n_demo = 9 + 3 + 3 + 3 + 6*3
        interval_demo_steps = 0.1

        n_scan = 0
        n_priv = 0
        n_priv_latent = 4 + 1 + 19*2
        n_state_label = 0
        n_dr_label = 0
        # n_proprio = 3 + 2 + 2 + 19*3 + 2# one hot


        # -------------------------- #
        history_len = 6
        n_priv_latent = 0
        n_priv = 0

        if task.motion_task == 'walk':
            n_proprio = 3 + 2 + 2 + 19*3 +3 -2 + 3
            # n_proprio = 3 + 2 + 2 + 19*3 +3 -2 + 3 + 132
            n_demo = 0
            episode_length_s = 14  # 20  18
        else: # recovery
            n_proprio = 3 + 2 + 2 + 19*3 +3 -2 + 3 
            n_demo = 0
            episode_length_s = 14
        
        if task.use_gait == True:
            n_proprio = n_proprio + 2

        if task.train_estimator:
            # Future state labels: base linear velocity, stability margin, base height
            n_state_label = 3 + 1 + 1
            # Domain randomization labels: mass, friction, motor strength, gains, push state
            n_dr_label = 4 + 1 + 19 * 2 + 19 * 2 + 5
            n_priv = n_state_label + n_dr_label
            n_latent = 16
    
        prop_hist_len = 5    # 4    10    20
        n_feature = prop_hist_len * n_proprio
        n_decoder_out = n_latent + n_priv

        num_observations = n_feature + n_proprio + n_priv

        n_priv_latent = n_dr_label
        n_priv_proprio = n_proprio + n_priv + n_priv_latent + 132 + 3 * 22 * 2
        n_priv_feature = n_priv_proprio * prop_hist_len
        num_privileged_obs = n_priv_feature + n_priv_proprio
        num_policy_actions = 19

    class motion:

        motion_curriculum = False

        motion_type = "yaml"
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
        static_friction = 0.6
        dynamic_friction = 0.6
        downsampled_scale = 0.15
        y_range = [-0.1, 0.1]

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
        curriculum = True
        measure_heights = True
        
        terrain_length = 20       # 20
        terrain_width = 16         #10 12 
        num_rows= 20 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 12 # number of terrain cols (types)

    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.05] # x,y,z [m] [0.0, 0.0, 1.1]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        lower_stand = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.30,         
           'left_knee_joint' : 0.65,       
           'left_ankle_joint' : -0.2,
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.30,                                       
           'right_knee_joint' : 0.65,                                             
           'right_ankle_joint' : -0.2,
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
           'left_shoulder_pitch_joint' : 0.1,  # 0.2
           'left_shoulder_roll_joint' : 0.1,   # 0.2
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 1.2,
           'right_shoulder_pitch_joint' : 0.1,  # 0.2
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

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/h1_custom_collision.urdf'
        torso_name = "torso_link"
        foot_name = "ankle"
        knee_name = "knee"    
        elbow_name = "elbow"
        waist_name = "torso_joint"

        # joint indices
        hip_joints = ["left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint"]
        # hip_joints = ["left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_roll_joint", "right_hip_yaw_joint"]
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_joint", "right_ankle_joint"]
        shoulder_joints = ['left_shoulder_roll_joint', 'left_shoulder_pitch_joint', 'left_shoulder_yaw_joint',
                          'right_shoulder_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_yaw_joint']
        elbow_joints = ['left_elbow_joint', 'right_elbow_joint']
        waist_joints = ['torso_joint']

        penalize_contacts_on = ["pelvis", "shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "hip"]

        # terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
        if task.motion_task == 'recovery':
            penalize_contacts_on = ["pelvis", "shoulder", "elbow", "hip", "knee"]
            # terminate_after_contacts_on = ["pelvis", "shoulder", "hip", "knee"]
            # terminate_after_contacts_on = ["pelvis", "hip", "knee"]
            # terminate_after_contacts_on = ["pelvis"]
            terminate_after_contacts_on = ["pelvis", "hip"]

        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # tracking rewards
            if task.motion_task == 'walk':
            #| ------------------------------------|
                # Task reward                
                joint_pos = 0.8      # 1.6    0.8   
                feet_clearance = 0.5     # 1.   0.5    0.4
                feet_contact_number = 1.0     # 1.2    0.6    0.5
              
                tracking_lin_vel = 2.0    #  2.0  1.2  1.2   5.0
                tracking_ang_vel = 2.0    #  1.0  0.8  1.2   5.0
                vel_mismatch_exp = 0.5  # lin_z; ang x,y
                # low_speed = 0.2
                # track_vel_hard = 0.5
                alive = 0.40      # 1  6  10

                # Regularization reward
                feet_distance = 0.2
                knee_distance = 0.2
                shoulder_pos = -1.0  # -1.0 
                elbow_pos = -0.5 # -1.0
                waist_pos = -1.0  # -1.0
                hip_pos = -1.0   # -4.0  -5.0
                orientation = -2.0        # 1.0  0.4  0.2  0.5  0.8

                # Feet reward
                feet_air_time = 2.0
                feet_stumble = -2
                feet_drag = -0.1
                feet_contact_forces = -0.01

                # Safe & Energy reward
                action_smoothness = -0.001  # -0.002
                torques = -5e-6   # -1e-5
                dof_vel = -5e-4   
                dof_acc = -5e-8   # -1e-7
                base_acc = 0.2
                collision = -10.0    # -1.
                dof_pos_limits = -10.0

            else:
                # recovery
                # feet_air_time = 1.     # 1.               
                feet_stumble = -2
                feet_drag = -0.1
                # joint_pos = 0.8      # 1.6    0.8   
                # feet_clearance = 0.5     # 1.   0.5    0.4
                # feet_contact_number = 1.0     # 1.2    0.6    0.5

                # Command tracking reward
                tracking_lin_vel = 1.2    # 1.0 10   0.8 1.2
                tracking_ang_vel = 1.0    # 0.5 5    0.4 1.0
                vel_mismatch_exp = 0.5  # lin_z; ang x,y   0.25

                feet_distance = 0.2
                knee_distance = 0.2
                
                lower_stand = -0.2
                shoulder_pos = -0.5  # -1.0 
                elbow_pos = -0.5 # -1.0
                waist_pos = -1.0  # -1.0
                hip_pos = -1.0   # -5.0
                # base_height = 0.4

                orientation = -2.0

                # Safe reward
                feet_contact_forces = -0.01
                action_smoothness = -0.001  # -0.002
                torques = -5e-6   # -1e-5
                dof_vel = -4e-5
                dof_acc = -5e-8   # -1e-7   
                base_acc = 0.2     
                collision = -1.0   # -40
                alive = 1.0     # 1  4  10
                termination = -100 
                dof_pos_limits = -10.0

                # ZMP is handled as an explicit constrained cost for recovery.
                zmp_distance = 0.0


            # # recovery
            # # # | -----------------------------------------------|
            # # Gait reward
            # # joint_pos = 0.8      # 1.6    0.8   
            # # feet_clearance = 0.4     # 1.   0.5    0.4
            # # feet_contact_number = 0.5     # 1.2    0.6    0.5
            # # feet_air_time = 1.     # 1.               
            # feet_stumble = -2
            # feet_drag = -0.1

            # # Command tracking reward
            # tracking_lin_vel = 1.0    # 1.0 10   0.8
            # tracking_ang_vel = 0.8    # 0.5 5    0.4
            # vel_mismatch_exp = 0.4  # lin_z; ang x,y   0.25
            # # low_speed = 0.1
            # # track_vel_hard = 0.25

            # # lin_vel_z = -0.2
            # # ang_vel_xy = -0.05

            # # Regularization reward
            # # feet_stumble = -2
            # # feet_drag = -0.1

            # feet_distance = 0.2
            # knee_distance = 0.2
            
            # # no_case2
            # # default_joint_pos = 0.4    # 0.5   0.2   
            # # upper_pos = 0.2       # 2.0  10.0   4.0    0.2
            # # lower_stand = 1.0   # 1.0

            # # no_case3
            # # default_joint_pos = 1.0    # 0.5 
            # upper_pos = 1.0       # 2.0  10.0   4.0    0.5
            # lower_stand = 1.0   # 1.0
            # hip_pos = -5.0

            # orientation = 0.8        # 1.0  0.4   0.2   0.5   0.6

            # # Safe reward
            # feet_contact_forces = -0.01
            # action_smoothness = -0.002
            # torques = -1e-5
            # dof_vel = -5e-4
            # dof_acc = -1e-7       
            # base_acc = 0.2     
            # collision = -20.0   # -40
            # alive = 10     # 1  4  10
            # termination = -100
            # # low_stand = -0.05     # -0.01  
            # dof_pos_limits = -10.0
            # zmp_distance = 0.5      # Is it too high?    0.4    



            # # Narrow Terrain 

            # # Task reward                
            # # joint_pos = 1.0      # 1.6    0.8   
            # feet_clearance = 0.8     # 1.   0.5    0.4
            # feet_contact_number = 1.0     # 1.2    0.6    0.5
            
            # tracking_lin_vel = 1.2    #  2.0  1.2
            # tracking_ang_vel = 1.2    #  1.0  0.8
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5
            # alive = 10      # 1  6  10

            # # Regularization reward
            # feet_distance = 0.2
            # knee_distance = 0.2
            # default_joint_pos = 0.4    # 0.5   0.2
            # upper_pos = 4.0       # 4.0   10.0
            # hip_pos = -5.0   # -4.0
            # orientation = 0.8        # 1.0  0.4  0.2  0.5
            # feet_height = -4.0
            # base_height = 0.5 


            # # Safe & Energy reward
            # # Feet reward
            # feet_air_time = 10.0
            # feet_stumble = -2
            # feet_drag = -0.1
            # feet_contact_forces = -0.01
            # action_smoothness = -0.002
            # torques = -1e-5
            # dof_vel = -5e-4
            # dof_acc = -1e-7
            # base_acc = 0.2
            # collision = -10.0    # -1.
            # dof_pos_limits = -10.0
            # termination = -100    # -100  -200      -20

            # # ZMP reward
            # # zmp_distance = 0.2      # 0.4   0.2
            # # angular_momentum
            # zmp_distance_exp = 1.0
            # angular_momentum = 2.0



            # # Perspective humanoid locomotion
            # # Task reward                
            # # joint_pos = 1.0      # 1.6    0.8   
            # feet_clearance = 0.8     # 1.   0.5    0.4
            # feet_contact_number = 1.0     # 1.2    0.6    0.5
            
            # tracking_lin_vel = 1.2    #  2.0  1.2
            # tracking_ang_vel = 1.2    #  1.0  0.8
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5
            # alive = 10      # 1  6  10

            # # Regularization reward
            # feet_distance = 0.2
            # knee_distance = 0.2
            # default_joint_pos = 0.4    # 0.5   0.2
            # upper_pos = 4.0       # 4.0   10.0
            # hip_pos = -5.0   # -4.0
            # orientation = 0.8        # 1.0  0.4  0.2  0.5
            # feet_height = -4.0
            # base_height = 0.5 

            # # Safe & Energy reward
            # # Feet reward
            # feet_air_time = 10.0
            # feet_stumble = -2
            # feet_drag = -0.1
            # feet_contact_forces = -0.01
            # action_smoothness = -0.002
            # torques = -1e-5
            # dof_vel = -5e-4
            # dof_acc = -1e-7
            # base_acc = 0.2
            # collision = -10.0    # -1.
            # dof_pos_limits = -10.0
            # termination = -100    # -100  -200      -20


        if task.motion_task == 'walk':
            tracking_sigma = 4.0  # 5.0
            base_height_target = 1.0    # 0.90 0.95
        else:
            tracking_sigma = 2.0
            base_height_target = 1.0    # 0.90 0.95
        # Use a geometric support polygon derived from the H1 model instead of a
        # hand-tuned center-distance threshold.
        stability_margin_threshold = 0.0
        use_zmp_cost = task.motion_task == 'recovery'
        zmp_cost_type = "margin"
        zmp_no_contact_cost = 0.0
        if task.motion_task == 'walk':
            zmp_margin_slack = 0.0
            zmp_cost_clip = 0.0
            zmp_single_support_weight = 1.0
            zmp_double_support_weight = 1.0
            zmp_com_vel_filter_alpha = 1.0
        else:
            # Recovery should still allow aggressive rescue steps, but it needs a
            # much tighter notion of "acceptable" edge support than before.
            zmp_margin_slack = 0.005
            zmp_cost_clip = 0.05
            zmp_single_support_weight = 0.8
            zmp_double_support_weight = 1.0
            zmp_com_vel_filter_alpha = 0.2
        # Support polygon parameters are derived from the H1 model:
        # - URDF left/right_foot_keypoint_joint is at x = 0.14 m from the ankle
        # - Unitree H1 official key dimensions list foot depth as 220 mm
        #   so the back extent is approximated as 0.22 - 0.14 = 0.08 m
        # - The ankle mesh span in y is about +/- 0.015 m
        support_polygon_front = 0.14
        support_polygon_back = 0.08
        support_polygon_sole_z = -0.04
        support_polygon_contact_tolerance = 0.004
        if task.motion_task == 'walk':
            support_polygon_left = 0.015
            support_polygon_right = 0.015
            support_polygon_shrink = 0.003
        else:
            # Use a noticeably wider polygon during recovery so ZMP only reacts to
            # clearly dangerous lateral excursions.
            support_polygon_left = 0.05
            support_polygon_right = 0.05
            support_polygon_shrink = 0.0
       
        # humanoid gym
        # max_contact_force = 1000
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06       # m
        target_feet_air_time = 0.18     # sec
        cycle_time = 0.65                # sec

        only_positive_rewards = False     # False
        clip_rewards = False        # True
        soft_dof_pos_limit = 0.98

        Gait_reward = [
            "joint_pos",           # 用于控制关节位置的奖励
            "feet_clearance",      # 确保步态中脚的清晰抬起
            "feet_contact_number" # 奖励正确的脚部接触数量
        ]

        # Gait_reward = []

        # For lower pos
        lower_pos = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.6,         
           'left_knee_joint' : 1.0,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.6,                                       
           'right_knee_joint' : 1.0,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0.1, 
           'left_shoulder_roll_joint' : 0.2, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 1.2,
           'right_shoulder_pitch_joint' : 0.1,
           'right_shoulder_roll_joint' : -0.2,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 1.2,
        }


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

        if task.motion_task == 'walk':
            push_robots = True
            push_interval_s = 8
            max_push_vel_xy = 1.0   # 0.5   1.0  2.0
            max_push_ang_vel = 0.5   # 0.3
            dynamic_randomization = 0.02
            max_force = 10
            max_torque = 10

        else:
            push_robots = True
            push_interval_s = 5     # 6
            max_push_vel_xy = 3.0   # 0.5   1.0  2.0
            max_push_ang_vel = 1.0   # 0.3
            dynamic_randomization = 0.02
            max_force = 20
            max_torque = 20

    class noise():
        add_noise = True
        noise_scale = 1.0 # scales other values    2.0 for recovery policy
        class noise_scales():
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
        curriculum = task.motion_task == 'recovery'
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before commands are resampled [s]
        if task.motion_task == 'recovery':
            resampling_time = 100.
        recovery_command_curriculum_steps = 24 * 5000
        heading_command = False # True   if true: compute ang vel command from heading error

        # For clip
        lin_vel_clip = 0.1     # default 0.2
        ang_vel_clip = 0.2     # default 0.4

        class recovery_curriculum_start_ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]


        class ranges:
            lin_vel_x = [-0.6, 2.5] # min max [m/s]   2.0
            lin_vel_y = [-0.6, 0.6]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-1.6, 1.6]
            
            # #Standing still
            # lin_vel_x = [0.0, 0.0] # min max [m/s]
            # lin_vel_y = [0.0, 0.0]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            # ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            # heading = [-1.6, 1.6]

            # # For deployment demo    Only forward
            # lin_vel_x = [0.0, 1.2] # min max [m/s]   2.0
            # lin_vel_y = [-0.1, 0.1]#[0.15, 0.6]   # min max [m/s]   # After depolying, spread it.
            # ang_vel_yaw = [-0.1, 0.1]    # min max [rad/s]
            # heading = [-1.6, 1.6]

class H1CommandCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimic"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPOMimic'
    
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = True

        text_feat_input_dim = H1CommandCfg.env.n_feature

        text_feat_output_dim = 16
        feat_hist_len = H1CommandCfg.env.prop_hist_len


    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        schedule = 'adaptive' # could be adaptive, fixed
        use_zmp_cost = H1CommandCfg.rewards.use_zmp_cost
        # Margin cost is measured in meters of support-polygon violation.
        zmp_cost_limit = 0.004 # 0.12 0.03
        zmp_lambda_init = 0.0
        zmp_lambda_lr = 1.0e-2 # 1.0e-4
        zmp_lambda_max = 0.1
        zmp_cost_value_loss_coef = 0.1
        normalize_cost_advantages = False

    class estimator:
        train_with_estimated_states = True    #False
        learning_rate = 1.e-4
        future_horizon = 1

        # For Deployment
        encoder_hidden_dims = [256, 128, 64]
        decoder_hidden_dims = [256, 128, 64]

        # # For simulation
        # encoder_hidden_dims = [512, 256]
        # decoder_hidden_dims = [256, 128, 64]

        n_demo = H1CommandCfg.env.n_demo
        priv_latent_dim = H1CommandCfg.env.n_priv_latent

        history_len = H1CommandCfg.env.prop_hist_len
        priv_states_dim = H1CommandCfg.env.n_priv
        state_label_dim = H1CommandCfg.env.n_state_label
        dr_label_dim = H1CommandCfg.env.n_dr_label
        priv_start = H1CommandCfg.env.n_feature + H1CommandCfg.env.n_proprio
        prop_start = H1CommandCfg.env.n_feature
        prop_dim = H1CommandCfg.env.n_proprio
        priv_prop_start = H1CommandCfg.env.n_priv_feature


class H1MimicDistillCfgPPO( H1CommandCfgPPO ):
    class distill:
        num_demo = 3
        num_steps_per_env = 24
        
        num_pretrain_iter = 0

        activation = "elu"
        learning_rate = 1.e-4
        student_actor_hidden_dims = [1024, 1024, 512]
        num_mini_batches = 4
