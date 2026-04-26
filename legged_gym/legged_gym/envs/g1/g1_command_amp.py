from isaacgym.torch_utils import *
import torch
from legged_gym.utils.math import *
# from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.g1.g1_command import G1Command, global_to_local, local_to_global
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from motion_lib import MotionLib
from isaacgym import gymtorch, gymapi, gymutil
import torch_utils
import os

class G1CommandAMP(G1Command):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        
        self._num_amp_obs_per_step = cfg.amp.num_obs_per_step
        self._num_amp_obs_steps = cfg.amp.num_obs_steps
        self._amp_obs_buf = torch.zeros((cfg.env.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=sim_device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None

        
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
    
    def reset_idx(self, env_ids, init=False):
        super().reset_idx(env_ids, init)
        if len(env_ids) != 0:
            self._compute_amp_observations(env_ids)
            self._init_amp_obs_default(env_ids)
        return
    
    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations() # latest on the left

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        return
    

    
    def _compute_amp_observations(self, env_ids=None):
        cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim, :3], self.root_states[:, :3])#.view(self.num_envs, -1)
        # print("cur_key_body_pos_local", cur_key_body_pos_local.shape)
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations_curr(self.root_states[:, :3], self.base_quat, self.base_lin_vel, self.base_ang_vel,
                                                                self.dof_pos, self.dof_vel, cur_key_body_pos_local)
            # self._curr_amp_obs_buf[:] = build_amp_observations_curr(self.base_quat, self.base_lin_vel, self.base_ang_vel,
            #                                                     self.dof_pos, self.dof_vel, cur_key_body_pos_local)                                            
            # self._curr_amp_obs_buf[:] = build_amp_observations_curr(self.root_states[:, :3], self.base_quat, self.base_lin_vel, self.base_ang_vel,
            #                                         self.dof_pos, self.dof_vel)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations_curr(self.root_states[env_ids, :3], self.base_quat[env_ids], self.base_lin_vel[env_ids], self.base_ang_vel[env_ids],
                                                                     self.dof_pos[env_ids], self.dof_vel[env_ids],
                                                                     cur_key_body_pos_local[env_ids] )
            # self._curr_amp_obs_buf[env_ids] = build_amp_observations_curr(self.base_quat[env_ids], self.base_lin_vel[env_ids], self.base_ang_vel[env_ids],
            #                                                          self.dof_pos[env_ids], self.dof_vel[env_ids],
            #                                                          cur_key_body_pos_local[env_ids] )
            # self._curr_amp_obs_buf[env_ids] = build_amp_observations_curr(self.root_states[env_ids, :3], self.base_quat[env_ids], self.base_lin_vel[env_ids], self.base_ang_vel[env_ids],
            #                                                 self.dof_pos[env_ids], self.dof_vel[env_ids])
        return
    

    ######### demonstrations #########
    def fetch_amp_obs_demo(self, num_samples):
        

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = self._motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat
    
    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times, get_lbp=True)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        # print("local_key_body_pos",local_key_body_pos.shape)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos)
        # amp_obs_demo = build_amp_observations(root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos)
        # amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel)
        return amp_obs_demo
    

    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        return dof_pos, dof_vel

    ######### utils #########
    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
    
    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return
    
    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step


    # def init_motions(self, cfg):
    #     self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)
    #     self._key_body_ids_sim = torch.tensor([0, 3, 5, # Left Hip yaw, Knee, Ankle
    #                                            6, 9, 11,
    #                                            13, 16, 17, # Left Shoulder pitch, Elbow, hand
    #                                            18, 20, 21], device=self.device)

    #     self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
    #     # self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
    #     # self._key_body_ids_sim_subset = torch.tensor([0, 1, 3, 4, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
    #     self._num_key_bodies = len(self._key_body_ids_sim_subset)


    #     self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
    #                           4, 5, 6,
    #                           7,       # Torso
    #                           8, 9, 10, # Shoulder, Elbow, Hand
    #                           11, 12, 13]  # 13
    #     self._dof_offsets = [0, 3, 4, 6, 9, 10, 12, 
    #                             13, 
    #                             16, 17, 18, 21, 22, 23]  # 14
        
    #     self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+ 10, device=self.device, dtype=torch.bool)
    #     # self._valid_dof_body_ids[-1] = 0
    #     # self._valid_dof_body_ids[-6] = 0
        
    #     self.dof_indices_sim = torch.tensor([0, 1, 2, 6, 7, 8, 13, 14, 15, 18, 19, 20], device=self.device, dtype=torch.long)
    #     self.dof_indices_motion = torch.tensor([1, 0, 2, 7, 6, 8, 14, 13, 15, 19, 18, 20], device=self.device, dtype=torch.long)

    #     # self._dof_ids_subset = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
    #     # self._dof_ids_subset = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
    #     # self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no ankle
        
    #     self._dof_ids_subset = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no ankle
    #     self._n_demo_dof = len(self._dof_ids_subset)


    #     if cfg.motion.motion_type == "single":
    #         motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy_g1/{cfg.motion.motion_name}.npy")
    #     else:
    #         assert cfg.motion.motion_type == "yaml"
    #         motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")
        
    #     self._load_motion(motion_file, cfg.motion.no_keybody)
    

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

    def draw_rigid_bodies_demo(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)
        for i in range(global_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
    dof = dof.clone()
    dof[:, indices_sim] = dof[:, indices_motion]
    # print('dof', dof.shape)
    # print('valid_dof_body_ids', len(valid_dof_body_ids))
    return dof[:, valid_dof_body_ids]

# @torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
    local_root_vel = quat_rotate(heading_rot, root_vel)

    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

# # @torch.jit.script
# def build_amp_observations(root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
#     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
#     local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
#     local_root_vel = quat_rotate(heading_rot, root_vel)

#     roll, pitch, yaw = euler_from_quaternion(root_rot)
#     return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)



# # @torch.jit.script
# def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel):
#     heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
#     local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
#     local_root_vel = quat_rotate(heading_rot, root_vel)

#     roll, pitch, yaw = euler_from_quaternion(root_rot)
#     return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3]), dim=-1)


# @torch.jit.script
def build_amp_observations_curr(root_pos, root_rot, local_root_vel, local_root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

# # @torch.jit.script
# def build_amp_observations_curr(root_rot, local_root_vel, local_root_ang_vel, dof_pos, dof_vel, local_key_body_pos):
#     roll, pitch, yaw = euler_from_quaternion(root_rot)
#     return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)


# # @torch.jit.script
# def build_amp_observations_curr(root_pos, root_rot, local_root_vel, local_root_ang_vel, dof_pos, dof_vel):
#     roll, pitch, yaw = euler_from_quaternion(root_rot)
#     return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3]), dim=-1)