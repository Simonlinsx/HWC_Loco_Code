from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot_command import LeggedRobot_command, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from .lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch

# from rsl_rl.runners import OnPolicyRunnerMimic

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

from motion_lib import MotionLib
import torch_utils

class G1Command(LeggedRobot_command):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        # Simon: to save the obs demo when inferring
        self.obs_demo_save = []

        
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)

        # self.num_privileged_obs = self.cfg.env.priv_num_observations
        # ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint']

        if self.cfg.task.motion_task == 'recovery':
            self.extreme_data = np.load("../extreme_data/extrem_data_g1.npy", allow_pickle=True)

        self.train_estimator = self.cfg.task.train_estimator

        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.init_motions(cfg)

        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._init_foot()
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device = self.device)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos[:, :10])
        
        # push init
        self.rand_push_force = torch.zeros((self.num_envs, 2), device = self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), device = self.device)

        # zmp init
        self.initialize_zmp()

        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        # init low pass filter
        if self.cfg.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.cfg.env.num_envs*self.cfg.env.num_actions),
                                                        highcut=np.ones(self.cfg.env.num_envs*self.cfg.env.num_actions) * self.cfg.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.cfg.env.num_envs * self.cfg.env.num_actions, 
                                                        device=self.device)


        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()


    # def _get_noise_scale_vec(self, cfg):
    #     noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
    #     noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
    #     noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu

    #     noise_scale_vec[:, 5:5+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
    #     noise_scale_vec[:, 5+self.num_dof:5+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
    #     noise_scale_vec[:, 5+3*self.num_dof:8+3*self.num_dof] = self.cfg.noise.noise_scales.gravity

    #     return noise_scale_vec
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:6] = self.cfg.noise.noise_scales.gravity
        noise_scale_vec[:, 6:6+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 6+self.num_dof:6+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        

        return noise_scale_vec
    

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

        # Sixu: 
    def init_motions(self, cfg):
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)
        
        self._key_body_ids_sim = torch.tensor([3, 4, 6, # Left Hip yaw, Knee, Ankle
                                               9, 10, 12,
                                               14, 17, 18, # Left Shoulder pitch, Elbow, hand
                                               19, 22, 23], device=self.device)

        # self._key_body_ids_sim = torch.tensor([14, 17, 18, # Left Shoulder pitch, Elbow, hand
        #                                        19, 22, 23], device=self.device)

        # self._key_body_ids_sim = torch.tensor([3, 4, 6, # Left Hip yaw, Knee, Ankle
        #                                        13, 14, 16,
        #                                        22, 25, 26, # Left Shoulder pitch, Elbow, hand
        #                                        27, 30, 31], device=self.device)


        # self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._key_body_ids_sim_subset_whole_body = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle

        # self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        # self._key_body_ids_sim_subset = torch.tensor([0, 1, 3, 4, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        self._num_key_bodies_whole_body  = len(self._key_body_ids_sim_subset_whole_body)

        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13
        
        self._dof_offsets = [0, 3, 4, 6, 9, 10, 12, 
                                13, 
                                16, 17, 18, 21, 22, 23]  # 14
        
        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+ 10, device=self.device, dtype=torch.bool)
        # self._valid_dof_body_ids[-1] = 0
        # self._valid_dof_body_ids[-6] = 0
        
        # self.dof_indices_sim = torch.tensor([0, 1, 2, 6, 7, 8, 13, 14, 15, 18, 19, 20, 21, 22], device=self.device, dtype=torch.long)
        # self.dof_indices_motion = torch.tensor([1, 0, 2, 7, 6, 8, 14, 13, 15, 19, 18, 20, 21, 22], device=self.device, dtype=torch.long)

        self.dof_indices_sim = torch.tensor([0, 1, 2, 6, 7, 8, 13, 14, 15, 18, 19, 20], device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.tensor([1, 0, 2, 7, 6, 8, 14, 13, 15, 19, 18, 20], device=self.device, dtype=torch.long)


        # [0, 1, 2]  3 left hips
        # [3]        1 left knee
        # [4, 5]     2 left ankles
        # [6, 7, 8]  3 right hips
        # [9]        1 right knee
        # [10, 11]   2 right ankles
        # [12]       1 waist 
        # [13, 14, 15] 3 left shoulders
        # [16]       1 left elbow
        # [17]       1 left hand
        # [18, 19, 20] 3 right shoulders
        # [21]       1 right elbow
        # [22]       1 right hand
        
        # Track all the dof pos
        # self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # no ankle
        
        # # No ankle dof pos.  Mimic ankle is not necessary
        self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # no ankle
        
        # No lower body
        # self._dof_ids_subset = torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22], device=self.device)  # no ankle
        

        self._n_demo_dof = len(self._dof_ids_subset)

        
        if cfg.motion.motion_type == "single":
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy_g1/{cfg.motion.motion_name}.npy")
        else:
            assert cfg.motion.motion_type == "yaml"
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")

        # motion_file = '/mnt/data1/zhaohaoyu/Whole-body-control-main/ASE/ase/poselib/data/configs/motions_autogen_all_no_run_jump_g1.yaml'
        # motion_file = '/mnt/data1/zhaohaoyu/Whole-body-control-main/ASE/ase/poselib/data/configs/motions_debug_g1.yaml'
        # print('cfg.motion.motion_name',cfg.motion.motion_name)
        # print('motion_file',motion_file)
        self._load_motion(motion_file, cfg.motion.no_keybody)



    def _load_motion(self, motion_file, no_keybody=False):
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return
    
    def draw_zmp_pos(self):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(1, 0, 0))
        self.zmp_pos = gymapi.Transform(gymapi.Vec3(self.zmp_x[self.lookat_id], self.zmp_y[self.lookat_id], torch.tensor(0.0, device=self.device)), r=None)
        gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], self.zmp_pos)


    def initialize_zmp(self):
        self.weighted_position_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.weighted_velocity_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.last_com_vel = torch.zeros(self.num_envs, 3 , device=self.device)
        self.zmp_distance = torch.zeros(self.num_envs, 1 , device=self.device)


    def compute_zmp(self):
        total_mass = torch.zeros(self.num_envs, device=self.device)
        # total_mass = 0
        self.weighted_position_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.weighted_velocity_sum = torch.zeros(self.num_envs, 3 , device=self.device)

        # for i, body in enumerate(self.body_properties):
        for i in range(22):
            # 这里可能有点问题，最好改成对于每个环境单独计算，因为mass有randomization
            # mass = body.mass
            mass = self.mass_tensor[:, i]  # shape: [num_envs]
            
            position = self.rigid_body_states[:, i, 0:3]
            velocity = self.rigid_body_states[:, i, 7:10]

            self.weighted_position_sum[:, 0] += position[:, 0] * mass
            self.weighted_position_sum[:, 1] += position[:, 1] * mass
            self.weighted_position_sum[:, 2] += position[:, 2] * mass

            self.weighted_velocity_sum[:, 0] += velocity[:, 0] * mass
            self.weighted_velocity_sum[:, 1] += velocity[:, 1] * mass
            self.weighted_velocity_sum[:, 2] += velocity[:, 2] * mass
            total_mass +=mass

        total_mass = total_mass.view(self.num_envs, 1)  # Shape: [num_envs, 1]
        # breakpoint()
        
        # The position of the central mass
        self.com_pos = torch.cat(
            (self.weighted_position_sum[:, 0].view(-1,1) / total_mass,
            self.weighted_position_sum[:, 1].view(-1,1) / total_mass,
            self.weighted_position_sum[:, 2].view(-1,1) / total_mass), -1
        ).view(self.num_envs, 3)

        # The position of the central mass
        com_vel = torch.cat(
            (self.weighted_velocity_sum[:, 0].view(-1,1)  / total_mass,
            self.weighted_velocity_sum[:, 1].view(-1,1)  / total_mass,
            self.weighted_velocity_sum[:, 2].view(-1,1)  / total_mass), -1
        ).view(self.num_envs, 3)

        # Step 1: Determine contact status for both feet in each environment
        # `contact_status` has shape [num_envs, 2] where each entry is True/False
        contact_status = self.contact_filt # Shape: [num_envs, 2]
        # print(contact_status)

        denom = torch.sum(contact_status, dim=1)

        # print(self.contact_filt)
        measured_heights = torch.sum(
            self.rigid_body_states[:, self.feet_indices, 2] * contact_status, dim=1) / torch.sum(contact_status, dim=1)

        measured_heights[denom == 0] = 0.0

        self.com_pos[:, 2] = self.com_pos[:, 2] - measured_heights - 0.07

        # Step 1: Calculate ZMP position for each environment
        com_acc = (com_vel - self.last_com_vel) / self.dt

        self.zmp_x = self.com_pos[:,0] - (self.com_pos[:,2] / 9.81) * com_acc[:, 0]
        self.zmp_y = self.com_pos[:,1] - (self.com_pos[:,2] / 9.81) * com_acc[:, 1]

        self.last_com_vel = com_vel 

        # Step 2: Get the (x, y) positions of both feet for each environment
        feet_xy = self.rigid_body_states[:, self.feet_indices, :2] # Shape: [num_envs, 2, 2]

        # Step 3: Initialize support center tensor for each environment
        support_center = torch.zeros((feet_xy.shape[0], 2), device=feet_xy.device) # Shape: [num_envs, 2]

        # Step 4: Calculate the support center based on contact conditions
        # Single support (left foot only)
        left_support_mask = (contact_status[:, 0]) & (~contact_status[:, 1]) # Shape: [num_envs]
        support_center[left_support_mask] = feet_xy[left_support_mask, 0, :]

        # Single support (right foot only)
        right_support_mask = (~contact_status[:, 0]) & (contact_status[:, 1])
        support_center[right_support_mask] = feet_xy[right_support_mask, 1, :]

        # Double support (both feet)
        double_support_mask = contact_status[:, 0] & contact_status[:, 1]
        support_center[double_support_mask] = (feet_xy[double_support_mask, 0, :] + feet_xy[double_support_mask, 1, :]) / 2.0

        # # No contact mask
        # no_contact_mask = ~(contact_status[:, 0] | contact_status[:, 1])

        # # Step 5: Calculate the ZMP distance from the support center for each environment
        zmp_position = torch.stack((self.zmp_x, self.zmp_y), dim=-1) # Shape: [num_envs, 2]
        
        # self.zmp_distance = torch.norm(zmp_position - support_center, dim=-1) # Euclidean distance for each environment
        # self.zmp_distance[no_contact_mask] = 0.0

        # Update zmp_distance only for environments with contact
        has_contact_mask = contact_status[:, 0] | contact_status[:, 1]
        self.zmp_distance[has_contact_mask, 0] = torch.norm(
            zmp_position[has_contact_mask, :] - support_center[has_contact_mask, :], dim=-1
        )
        # Output the ZMP distance for each environment
        # print("ZMP Distance from Support Center for each environment:", self.zmp_distance)

    
    def step(self, actions):
        actions = self.reindex(actions).to(self.device)
        actions.to(self.device)
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        # if self.global_counter % 10 ==0:
        #     actions[:,7] = 10
        #     actions[:,3] = 10
        # else:
        #     actions[:,7] = 10
        #     actions[:,3] = 10

        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            # self.delay = torch.randint(0, 3, (1,), device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms


        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.global_counter += 1
        self.total_env_steps_counter += 1


        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        # clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)


        self.render()
        # self.actions[:, [4, 9]] = torch.clamp(self.actions[:, [4, 9]], -0.5, 0.5)
        # self.actions[:, [4, 10]] = torch.clamp(self.actions[:, [4, 10]], -0.5, 0.5)
        # print(self.actions[self.lookat_id, [4, 9]])
        self.actions[:, [5, 11]] = torch.clamp(self.actions[:, [5, 11]], -0.5, 0.5) # Clamp ankle roll's action
        # self.actions[:, [4, 5, 10, 11]] = torch.clamp(self.actions[:, [4, 5, 10, 11]], -0.5, 0.5) # Clamp ankle roll's action
        
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # print(self.torques[:, [4, 5, 10, 11]])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for i in torch.topk(self.torques[self.lookat_id], 3).indices.tolist():
        #     print(self.dof_names[i], self.torques[self.lookat_id][i])
        
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # def resample_motion_times(self, env_ids):
    #     return self._motion_lib.sample_time(self._motion_ids[env_ids])


    # def update_motion_ids(self, env_ids):
    #     self._motion_times[env_ids] = self.resample_motion_times(env_ids)
    #     self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
    #     self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])


    def domain_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
            

        if self.cfg.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.cfg.env.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.cfg.env.num_actions), device=self.device)
    

    def reset_extreme(self, env_ids):

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :1] += torch_rand_float(-1.0, 15.0, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
        self.root_states[env_ids, 1:2] += torch_rand_float(-2.0, 2.0, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
        terrain_height = self._get_heights()
        self.root_states[env_ids, 2] += terrain_height[env_ids, 66]
        self.initial_origins[env_ids, :3] = self.root_states[env_ids, :3]

        # 3 + 3 + 1 + 1 + 19 + 19 + 3  = 49 
        # [base_lin_vel, env.base_ang_vel, torch.stack((env.roll, env.pitch), dim = 1), env.dof_pos, env.dof_vel, env.commands[:, :3]]
        # self.extreme_data = np.load("/home/simon/expressive-humanoid/legged_gym/legged_gym/scripts/extrem_data.npy", allow_pickle=True)

        batch_size = len(env_ids)
        indices = np.random.choice(len(self.extreme_data), batch_size, replace=False)
        # sampled_data = np.array([data[i] for i in indices])
        sampled_data = torch.tensor([self.extreme_data[i] for i in indices]).to(self.device)

        self.root_states[env_ids, 7:10] = sampled_data[:, :3]
        self.root_states[env_ids, 10:13] = sampled_data[:, 3:6]

        rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        rand_roll = sampled_data[:, 6]
        rand_pitch = sampled_data[:, 7]
        # print(rand_pitch.shape)
        quat = quat_from_euler_xyz(rand_roll, rand_pitch, rand_yaw) 
        self.root_states[env_ids, 3:7] = quat[:, :] 
        
        self.dof_pos[env_ids] = sampled_data[:, 8:31]
        self.dof_pos[env_ids] = sampled_data[:, 31:54]
        self.commands[env_ids, :3] = sampled_data[:, 54:57]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
        
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        if self.cfg.env.extreme_flag == True and self.cfg.task.motion_task == 'recovery':
            flag = np.random.rand()
            if flag > 0.02:   # 0.1
                self._reset_dofs(env_ids)
                self._reset_root_states(env_ids)
                self._resample_commands(env_ids)  # no resample commands
            else:
                self.reset_extreme(env_ids)
        else:
            # reset robot states
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
            self._resample_commands(env_ids)  # no resample commands


        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)



        self.domain_randomization(env_ids)


        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.action_history_buf[env_ids, :, :] = 0.
        # fill extras
        self.extras["episode"] = {}
        # self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0


        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return

   
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                # self.root_states[env_ids, :2] += torch_rand_float(-1, 1, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
                self.root_states[env_ids, :1] += torch_rand_float(-0.01, 0.01, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
                self.root_states[env_ids, 1:2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
                terrain_height = self._get_heights()
                self.root_states[env_ids, 2] += terrain_height[env_ids, 66]
            
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            # if self.cfg.env.randomize_start_y:
            #     self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)

            # recovery
            if self.cfg.env.rand_vel:
                self.root_states[env_ids, 7:8] = torch_rand_float(self.cfg.env.rand_lin_x_vel[0], self.cfg.env.rand_lin_x_vel[1], \
                                                                        (len(env_ids), 1), device=self.device)  # [7:10]: lin vel
                self.root_states[env_ids, 8:9] = torch_rand_float(self.cfg.env.rand_lin_y_vel[0], self.cfg.env.rand_lin_y_vel[1], \
                                                                        (len(env_ids), 1), device=self.device)  # [7:10]: lin vel
                self.root_states[env_ids, 9:10] = torch_rand_float(self.cfg.env.rand_lin_z_vel[0], self.cfg.env.rand_lin_z_vel[1], \
                                                                        (len(env_ids), 1), device=self.device)  # [7:10]: lin vel
                self.root_states[env_ids, 10:13] = torch_rand_float(self.cfg.env.rand_ang_vel[0], self.cfg.env.rand_ang_vel[1], \
                                                                        (len(env_ids), 3), device=self.device)  # [10:13]: ang vel
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # self.initial_origins[env_ids, :3] = self.root_states[env_ids, :3]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.task.motion_task == 'walk':
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs), device=self.device)
            self.dof_vel[env_ids] = 0.
        else:   # recovery
            # self.dof_pos[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dofs), device=self.device)
            # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.2, 1.8, (len(env_ids), self.num_dofs), device=self.device)
            random_factors = torch_rand_float(0.2, 1.8, (len(env_ids), self.num_dofs), device=self.device)
            zero_random = torch_rand_float(-0.4, 0.4, (len(env_ids), self.num_dofs), device=self.device)
            self.dof_pos[env_ids] = self.default_dof_pos * random_factors + (self.default_dof_pos == 0).float() * zero_random
            # print(self.dof_pos[0])

            self.dof_vel[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dofs), device=self.device)
            # self.dof_vel[env_ids] = 0.
        
        # print(f"env_ids shape: {env_ids.shape}")
        # print(f"self.dof_state shape: {self.dof_state.shape}")
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



    def post_physics_step(self):
        super().post_physics_step()


        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self.draw_rigid_bodies_demo()
            # self.draw_rigid_bodies_actual()
            # self.draw_zmp_pos()

        return

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()


    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)

    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """

        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel

        self.rand_push_force = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_obs_buf_commands(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        # print(self.commands[:3,:3])
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            self.projected_gravity,
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.last_actions
                            ),dim=-1)


    def frequency_encoding(self, zmp_feature, F):
        """
        对 zmp 距离特征进行频率编码
        :param zmp_feature: 输入的 zmp 距离特征张量
        :param F: 频率编码的频率数量
        :return: 频率编码后的张量
        """
        encoding = []
        for i in range(F):
            freq = 2 ** i
            encoding.append(torch.sin(freq * torch.pi * zmp_feature))
            encoding.append(torch.cos(freq * torch.pi * zmp_feature))
        return torch.cat(encoding, dim=-1)
    
    
    def compute_observations(self):
        # self.zmp_distance
        self.compute_zmp()

        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)#self._demo_obs_buf[:, 2:, :].clone().flatten(start_dim=1) 
        priv_motion_features = self.priv_obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)

        # self.zmp = torch.log(self.zmp_distance + 1)
        heights = torch.clip(self.measured_heights, -1., 1.)

        measured_heights = torch.sum(self.rigid_body_states[:, self.feet_indices, 2], dim=1) / 2.0
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)

        # For encoding zmp
        zmp = self.frequency_encoding(self.zmp_distance, 4)
        # priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, zmp), dim=-1)

        # Base linear velocity zmp, and base height
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, zmp, base_height.unsqueeze(1)), dim=-1)

        priv_latent = torch.cat((      # dimensions
            self.mass_params_tensor,        # 4
            self.friction_coeffs_tensor,    # 1
            self.motor_strength[0] - 1,     # 29
            self.motor_strength[1] - 1,     # 29
            self._kp_scale,    # 29
            self._kd_scale,    # 29
            self.rand_push_force,    #  2
            self.rand_push_torque,    # 3
            heights    # 132
        ), dim=-1)


        obs_buf = self.compute_obs_buf_commands()
        self.command_input = self.commands[:, :3]
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        obs_buf = torch.cat((obs_buf, self.command_input, sin_phase, cos_phase), dim = -1)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale
        
        priv_obs_buf = torch.cat([obs_buf, priv_latent, priv_explicit], dim=-1)
        self.privileged_obs_buf = torch.cat([priv_motion_features, priv_obs_buf], dim=-1)

        if self.train_estimator == True:
            self.obs_buf = torch.cat([motion_features, obs_buf, priv_explicit], dim=-1)
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.priv_obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.priv_obs_history_buf[:, 1:],
                priv_obs_buf.unsqueeze(1)
            ], dim=1)
        )



    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # roll_cutoff = torch.abs(self.roll) > 1.0
        # pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5

        # print(self.roll, self.pitch)
        roll_cutoff = torch.abs(self.roll) > 1.0
        pitch_cutoff = torch.abs(self.pitch) > 1.0
        height_cutoff = self.root_states[:, 2] < 0.5
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff


    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        # self.ref_dof_pos = torch.zeros_like(self.dof_pos[:, :10])
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_l[torch.abs(sin_pos_l) < 0.1] = 0
        self.ref_dof_pos[:, 2] =  sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_hip_pitch_joint']
        self.ref_dof_pos[:, 3] =  sin_pos_l * scale_2 + self.cfg.init_state.default_joint_angles['left_knee_joint']
        self.ref_dof_pos[:, 4] =  sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_ankle_joint']
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        sin_pos_r[torch.abs(sin_pos_r) < 0.1] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r * scale_1 - self.cfg.init_state.default_joint_angles['right_hip_pitch_joint']
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2 - self.cfg.init_state.default_joint_angles['right_knee_joint']
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_1 - self.cfg.init_state.default_joint_angles['right_ankle_joint']
        # # Double support phase
        # self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos


    # ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            
            name = self.reward_names[i]
            # print(name)
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew #if "demo" not in name else 0  # log demo rew but do not include in additative reward
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

     

    # def compute_reward(self):
    #     self.rew_buf[:] = 0.  # 初始化奖励缓冲区
    #     zmp = torch.log(self.zmp_distance + 1)  # 计算零力矩点的距离
    #     rew_mask = (zmp <= 2.0).float().squeeze()  # 掩码：当 zmp <= 2.0 时为 1，否则为 0

        
    #     for i in range(len(self.reward_functions)):
    #         name = self.reward_names[i]
    #         rew = self.reward_functions[i]() * self.reward_scales[name]
    #         # print('rew_mask', rew_mask.shape)
    #         # print('rew', rew.shape)
    #         # 根据奖励类别应用掩码
    #         if name in self.cfg.rewards.Gait_reward + self.cfg.rewards.Command_tracking_reward + self.cfg.rewards.Upper_reward:
    #             rew *= rew_mask  # 将奖励置零

    #         self.rew_buf += rew  # 累加奖励
    #         self.episode_sums[name] += rew  # 累加到对应类别的总和

    #     if self.cfg.rewards.only_positive_rewards:
    #         self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    #     if self.cfg.rewards.clip_rewards:
    #         self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
    #     # add termination reward after clipping
    #     if "termination" in self.reward_scales:
    #         rew = self._reward_termination() * self.reward_scales["termination"]
    #         self.rew_buf += rew
    #         self.episode_sums["termination"] += rew






    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos[:, :10].clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        # diff[: , 4] = 0 * diff[: , 4]
        # diff[: , 9] = 0 * diff[: , 9]
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_body_states[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_elbow_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        elbow_pos = self.rigid_body_states[:, self.elbow_indices, :2]
        elbow_dist = torch.norm(elbow_pos[:, 0, :] - elbow_pos[:, 1, :], dim=1)
        # fd = self.cfg.rewards.min_dist
        # max_df = self.cfg.rewards.max_dist / 2
        # d_min = torch.clamp(elbow_dist - fd, -0.5, 0.)
        # d_max = torch.clamp(elbow_dist - max_df, 0, 0.5)
        # TODO need to change to 0.45?
        rew = torch.minimum(elbow_dist, torch.tensor(0.40, device=self.device))
        return rew

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    # def _reward_feet_air_time(self):
    #     """
    #     Calculates the reward for feet air time, promoting longer steps. This is achieved by
    #     checking the first contact with the ground after being in the air. The air time is
    #     limited to a maximum value for reward calculation.
    #     """
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    #     # stance_mask = self._get_gait_phase()
    #     # self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
    #     self.contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * self.contact_filt
    #     self.feet_air_time += self.dt
    #     air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
    #     self.feet_air_time *= ~self.contact_filt
    #     return air_time.sum(dim=1)



    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    # def _reward_orientation(self):
    #     """
    #     Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
    #     from the desired base orientation using the base euler angles and the projected gravity vector.
    #     """
    #     quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
    #     orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
    #     return (quat_mismatch + orientation) / 2.

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)


    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - 1000.0).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        # joint_diff = self.dof_pos - self.default_dof_pos
        # # joint_diff = self.dof_pos[:, :10] - self.default_dof_pos[:, :10]
        # left_yaw_roll = joint_diff[:, 1:3]
        # right_yaw_roll = joint_diff[:, 7:9]    
        # yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        # yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        # return torch.exp(-yaw_roll * 10) - 0.01 * torch.norm(joint_diff, dim=1)

        # joint_diff = self.dof_pos - self.default_dof_pos
        # joint_diff = self.dof_pos[:, :12] - self.default_dof_pos[:, :12]
        # [1,2,7,8,12]
        joint_diff = self.dof_pos[:, [1,2,7,8,12]] - self.default_dof_pos[:, [1,2,7,8,12]]
        # joint_diff = self.dof_pos[:, [1,2,4,5,7,8,10,11,12]] - self.default_dof_pos[:, [1,2,4,5,7,8,10,11,12]]
        return - torch.norm(joint_diff, dim=1)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res



    def _reward_upper_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        shoulder_pitch_diff = torch.abs(self.dof_pos[:, 13] - self.default_dof_pos[:, 13]) + torch.abs(self.dof_pos[:, 18]  - self.default_dof_pos[:, 18])
        shoulder_roll_diff = torch.abs(self.dof_pos[:, 14] - self.default_dof_pos[:, 14]) + torch.abs(self.dof_pos[:, 19]  - self.default_dof_pos[:, 19])
        shoulder_yaw_diff = torch.abs(self.dof_pos[:, 15] - self.default_dof_pos[:, 15]) + torch.abs(self.dof_pos[:, 20] - self.default_dof_pos[:, 20])
        torso_diff = self.dof_pos[:, 12] - self.default_dof_pos[:, 12]
        wrist_diff = torch.abs(self.dof_pos[:, 17] - self.default_dof_pos[:, 17]) + torch.abs(self.dof_pos[:, 22] - self.default_dof_pos[:, 22])
        elbow_diff = torch.abs(self.dof_pos[:, 16] - self.default_dof_pos[:, 16])  + torch.abs(self.dof_pos[:, 21] - self.default_dof_pos[:, 21])

        return - torch.abs(torso_diff) - torch.abs(shoulder_roll_diff) - torch.abs(shoulder_yaw_diff) - 0.1 * torch.abs(elbow_diff) - 0.1 * torch.abs(wrist_diff) - 0.08 * torch.abs(shoulder_pitch_diff) 


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        # print("lower dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 0])
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # print("upper dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 1])
        return torch.sum(out_of_limits, dim=1)



    def _reward_zmp_distance(self):
        # print(self.zmp_distance.shape)
        zmp = torch.log(self.zmp_distance + 1.0)
        zmp_rew = torch.exp(-zmp)

        # print(penalty.shape)
        return zmp_rew.squeeze()

    def _reward_upper_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        shoulder_roll_diff = self.dof_pos[:, 12] + self.dof_pos[:, 16] - self.default_dof_pos[:, 12] - self.default_dof_pos[:, 16]
        joint_diff = self.dof_pos[:, 10:] - self.default_dof_pos[:, 10:]
        return - torch.norm(joint_diff, dim=1) - torch.abs(shoulder_roll_diff)

    def _reward_lower_stand(self):
        # diff = self.dof_pos - self.lower_pos
        diff = self.dof_pos[:, :12] - self.default_dof_pos[:, :12]
        rew = - 0.1 * torch.norm(diff, dim=-1)
        return rew
    

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_feet_height(self):
        terrain_height = self._get_heights()
        feet_height = self.rigid_body_states[:, self.feet_indices, 2] - terrain_height[:, 66].unsqueeze(1)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
        height_diff = torch.square(feet_height - 0.1) * ~ contact
        rew = torch.sum(height_diff, dim=(1))

        mask = torch.norm(self.commands[:, :2], dim=1) < 0.1
        rew[mask] = 0.0
        return rew    

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_alive(self):
        return 1.0

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_feet_drag(self):
        # print(contact_bool)
        # contact_forces = self.contact_forces[:, self.feet_indices, 2]
        # print(contact_forces[self.lookat_id], self.force_sensor_tensor[self.lookat_id, :, 2])
        # print(self.contact_filt[self.lookat_id])
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:9]).sum(dim=-1)
        dragging_vel = self.contact_filt * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        # print(rew[self.lookat_id].cpu().numpy(), self.contact_filt[self.lookat_id].cpu().numpy(), feet_xy_vel[self.lookat_id].cpu().numpy())
        return rew


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()


    # def _reward_feet_parallel(self):
    #     left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
    #     right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
    #     feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
    #     feet_distances_var = torch.var(feet_distances, dim=1)
    #     # return feet_distances_var * (self.commands[:, 4] >= 0.735)
    #     return feet_distances_var

    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)
    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))
    return local_pos_delta.squeeze(-1)

