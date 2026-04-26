from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision
import torch.nn.functional as F

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

class H1Command(LeggedRobot_command):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.obs_demo_save = []
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.global_counter = 0
        self.total_env_steps_counter = 0
        self.command_curriculum_progress = 1.0
        self._recovery_command_target_ranges = {
            key: list(self.command_ranges[key])
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")
            if key in self.command_ranges
        }

        self.extreme_data = np.load("../extreme_data/extrem_data_paper.npy", allow_pickle=True)
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
        self._init_reset_reason_buffers()

        self._prepare_reward_function()
        self.init_done = True

        # init low pass filter
        if self.cfg.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.cfg.env.num_envs*self.cfg.env.num_actions),
                                                        highcut=np.ones(self.cfg.env.num_envs*self.cfg.env.num_actions) * self.cfg.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.cfg.env.num_envs * self.cfg.env.num_actions, 
                                                        device=self.device)
        self.lower_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.lower_stand[name]
            self.lower_pos[i] = angle

        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device = self.device)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos[:, :10])
        self.rand_push_force = torch.zeros((self.num_envs, 2), device = self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), device = self.device)


        self.continuous_force = torch.zeros(self.num_envs * len(self._body_list),  3,  device=self.device)
        self.continuous_torque = torch.zeros(self.num_envs * len(self._body_list),  3,  device=self.device)

        
        self.initialize_zmp()
        self._update_recovery_command_curriculum_ranges()
        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()

    def _init_reset_reason_buffers(self):
        self.reset_reason_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_reason_pelvis_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_reason_hip_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_reason_knee_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_reason_roll = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_reason_pitch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_reason_timeout = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        termination_contact_names = getattr(self, "termination_contact_names", [])

        def build_contact_mask(token):
            return torch.tensor(
                [token in name for name in termination_contact_names],
                dtype=torch.bool,
                device=self.device,
            )

        self._termination_contact_group_masks = {
            "pelvis": build_contact_mask("pelvis"),
            "hip": build_contact_mask("hip"),
            "knee": build_contact_mask("knee"),
        }

    def _reset_reason_from_contact_group(self, contact_mask, token):
        group_mask = self._termination_contact_group_masks[token]
        if group_mask.numel() == 0 or not torch.any(group_mask):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return torch.any(contact_mask[:, group_mask], dim=1).float()

    def _update_recovery_command_curriculum_ranges(self):
        progress = 1.0
        if (
            self.cfg.task.motion_task == "recovery"
            and getattr(self.cfg.commands, "curriculum", False)
            and hasattr(self.cfg.commands, "recovery_curriculum_start_ranges")
            and len(self._recovery_command_target_ranges) != 0
        ):
            warmup_steps = max(int(getattr(self.cfg.commands, "recovery_command_curriculum_steps", 1)), 1)
            progress = min(float(self.total_env_steps_counter) / float(warmup_steps), 1.0)
            start_ranges = self.cfg.commands.recovery_curriculum_start_ranges
            for key, target_range in self._recovery_command_target_ranges.items():
                start_range = list(getattr(start_ranges, key))
                self.command_ranges[key] = [
                    start_range[0] + (target_range[0] - start_range[0]) * progress,
                    start_range[1] + (target_range[1] - start_range[1]) * progress,
                ]
        elif len(self._recovery_command_target_ranges) != 0:
            for key, target_range in self._recovery_command_target_ranges.items():
                self.command_ranges[key] = list(target_range)
        self.command_curriculum_progress = progress
        return progress


    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, 5:5+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, 5+self.num_dof:5+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        noise_scale_vec[:, 5+3*self.num_dof:8+3*self.num_dof] = self.cfg.noise.noise_scales.gravity
        return noise_scale_vec
    
    def init_motions(self, cfg):
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)
        self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
                                               6, 9, 10,
                                               12, 15, 16, # Left Shoulder pitch, Elbow, hand
                                               17, 20, 21], device=self.device)
        
        self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13
        self._dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
                             11, 
                             14, 15, 16, 19, 20, 21]  # 14


    def _load_motion(self, motion_file, no_keybody=False):
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return

    def initialize_zmp(self):
        self.weighted_position_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.weighted_velocity_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.last_com_vel = torch.zeros(self.num_envs, 3 , device=self.device)
        self.filtered_com_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_filtered_com_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.zmp_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.zmp_distance = torch.zeros(self.num_envs, 1 , device=self.device)
        self.stability_margin = torch.zeros(self.num_envs, 1, device=self.device)
        self.zmp_valid_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cost_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_cost_sums = torch.zeros(self.num_envs, device=self.device)
        self._foot_forward_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self._foot_lateral_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._foot_vertical_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.support_plane_height = torch.zeros(self.num_envs, device=self.device)
        self.support_corners_world = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            4,
            3,
            device=self.device,
        )
        self.active_support_corner_mask = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            4,
            dtype=torch.bool,
            device=self.device,
        )

    def _create_envs(self):
        super()._create_envs()
        if self.cfg.env.record_video or self.cfg.env.record_frame:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 720
            camera_props.height = 480
            self._rendering_camera_handles = []
            for i in range(self.num_envs):
                camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
                self._rendering_camera_handles.append(camera_handle)
                cam_pos = np.array([2.0, 0.0, 0.3])
                self.gym.set_camera_location(
                    camera_handle,
                    self.envs[i],
                    gymapi.Vec3(*cam_pos),
                    gymapi.Vec3(*(0.0 * cam_pos)),
                )

    def render_record(self, mode="rgb_array"):
        if self.global_counter % 2 != 0:
            return None

        camera_states = []
        for i in range(self.num_envs):
            root_pos = self.root_states[i, :3].detach().cpu().numpy()
            cam_pos = root_pos + np.array([0.0, -2.0, 0.3], dtype=np.float32)
            cam_target = root_pos.astype(np.float32)
            cam = self._rendering_camera_handles[i]
            self.gym.set_camera_location(
                cam,
                self.envs[i],
                gymapi.Vec3(*cam_pos),
                gymapi.Vec3(*cam_target),
            )
            camera_states.append((cam, cam_pos, cam_target))

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        imgs = []
        for i, (cam, cam_pos, cam_target) in enumerate(camera_states):
            img = self.gym.get_camera_image(self.sim, self.envs[i], cam, gymapi.IMAGE_COLOR)
            h, w = img.shape
            frame = np.ascontiguousarray(img.reshape([h, w // 4, 4]))

            support_polygon, support_color = self._get_support_polygon_world(i)
            if support_polygon is not None:
                polygon_pixels = []
                for point_world in support_polygon:
                    pixel = self._project_world_to_pixel(i, cam, cam_pos, cam_target, point_world, frame.shape)
                    if pixel is not None:
                        polygon_pixels.append(pixel)
                if len(polygon_pixels) >= 2:
                    self._draw_overlay_polyline(frame, polygon_pixels, color=support_color, thickness=3)

            zmp_world = self._get_zmp_world_point(i)
            if zmp_world is not None:
                pixel = self._project_world_to_pixel(i, cam, cam_pos, cam_target, zmp_world, frame.shape)
                if pixel is not None:
                    self._draw_overlay_circle(frame, pixel, radius=6)

            imgs.append(frame)
        return imgs

    def _get_zmp_world_point(self, env_id):
        if not hasattr(self, "zmp_xy") or not self.zmp_valid_mask[env_id]:
            return None

        return np.array([
            float(self.zmp_xy[env_id, 0].item()),
            float(self.zmp_xy[env_id, 1].item()),
            float(self.support_plane_height[env_id].item()),
        ], dtype=np.float32)

    def _get_support_polygon_world(self, env_id):
        if not hasattr(self, "contact_filt") or not hasattr(self, "active_support_corner_mask"):
            return None, None

        contact = self.contact_filt[env_id]
        if not torch.any(contact):
            return None, None

        if contact[0] and not contact[1]:
            polygon_points = self.support_corners_world[env_id, 0, self.active_support_corner_mask[env_id, 0]]
            polygon_color = (255, 196, 0, 255)
        elif contact[1] and not contact[0]:
            polygon_points = self.support_corners_world[env_id, 1, self.active_support_corner_mask[env_id, 1]]
            polygon_color = (0, 196, 255, 255)
        else:
            polygon_points = torch.cat(
                (
                    self.support_corners_world[env_id, 0, self.active_support_corner_mask[env_id, 0]],
                    self.support_corners_world[env_id, 1, self.active_support_corner_mask[env_id, 1]],
                ),
                dim=0,
            )
            polygon_color = (0, 255, 0, 255)
        if polygon_points.shape[0] < 2:
            return None, None
        polygon_xy = polygon_points[:, :2]
        if polygon_xy.shape[0] >= 3:
            edge_starts, _, edge_mask = self._batched_convex_hull_edges(polygon_xy.unsqueeze(0))
            polygon_xy = edge_starts[0, edge_mask[0]]
            if polygon_xy.shape[0] < 2:
                return None, None
        polygon_z = torch.full(
            (polygon_xy.shape[0], 1),
            self.support_plane_height[env_id],
            dtype=polygon_xy.dtype,
            device=polygon_xy.device,
        )
        polygon_world = torch.cat((polygon_xy, polygon_z), dim=1)
        return polygon_world.detach().cpu().numpy().astype(np.float32), polygon_color

    def _project_world_to_pixel(self, env_id, cam_handle, cam_pos, cam_target, point_world, image_shape):
        forward = cam_target - cam_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1.0e-6:
            return None
        forward = forward / forward_norm

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(forward, world_up)) > 0.98:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1.0e-6:
            return None
        right = right / right_norm
        up = np.cross(right, forward)
        up = up / max(np.linalg.norm(up), 1.0e-6)

        rel = point_world - cam_pos
        cam_x = float(np.dot(rel, right))
        cam_y = float(np.dot(rel, up))
        cam_z = float(np.dot(rel, forward))
        if cam_z <= 1.0e-6:
            return None

        proj = np.array(
            self.gym.get_camera_proj_matrix(self.sim, self.envs[env_id], cam_handle),
            dtype=np.float32,
        ).reshape(4, 4)
        ndc_x = proj[0, 0] * cam_x / cam_z
        ndc_y = proj[1, 1] * cam_y / cam_z
        if abs(ndc_x) > 1.0 or abs(ndc_y) > 1.0:
            return None

        height, width = image_shape[:2]
        px = int((ndc_x + 1.0) * 0.5 * (width - 1))
        py = int((1.0 - ndc_y) * 0.5 * (height - 1))
        if px < 0 or px >= width or py < 0 or py >= height:
            return None
        return (px, py)

    def _draw_overlay_circle(self, frame, pixel, radius=6):
        cv2.circle(frame, pixel, radius + 2, (255, 255, 255, 255), thickness=-1)
        # `frame` is later encoded as RGBA, so use RGBA byte order here.
        cv2.circle(frame, pixel, radius, (255, 0, 0, 255), thickness=-1)

    def _draw_overlay_polyline(self, frame, pixels, color=(0, 255, 0, 255), thickness=3):
        if len(pixels) < 2:
            return
        pts = np.array(pixels, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _get_support_polygon_params(self):
        front = max(self.cfg.rewards.support_polygon_front - self.cfg.rewards.support_polygon_shrink, 1.0e-4)
        back = max(self.cfg.rewards.support_polygon_back - self.cfg.rewards.support_polygon_shrink, 1.0e-4)
        left = max(self.cfg.rewards.support_polygon_left - self.cfg.rewards.support_polygon_shrink, 1.0e-4)
        right = max(self.cfg.rewards.support_polygon_right - self.cfg.rewards.support_polygon_shrink, 1.0e-4)
        sole_z = getattr(self.cfg.rewards, "support_polygon_sole_z", -0.04)
        corner_tol = getattr(self.cfg.rewards, "support_polygon_contact_tolerance", 0.008)
        return front, back, left, right, sole_z, corner_tol

    def _get_feet_axes(self):
        feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7].reshape(-1, 4)
        batch = feet_quat.shape[0]
        foot_x = quat_apply(
            feet_quat,
            self._foot_forward_axis.unsqueeze(0).repeat(batch, 1),
        ).view(self.num_envs, len(self.feet_indices), 3)
        foot_y = quat_apply(
            feet_quat,
            self._foot_lateral_axis.unsqueeze(0).repeat(batch, 1),
        ).view(self.num_envs, len(self.feet_indices), 3)
        foot_z = quat_apply(
            feet_quat,
            self._foot_vertical_axis.unsqueeze(0).repeat(batch, 1),
        ).view(self.num_envs, len(self.feet_indices), 3)
        foot_x = F.normalize(foot_x, dim=-1, eps=1.0e-8)
        foot_y = F.normalize(foot_y, dim=-1, eps=1.0e-8)
        foot_z = F.normalize(foot_z, dim=-1, eps=1.0e-8)
        return foot_x, foot_y, foot_z

    def _get_feet_axes_xy(self):
        foot_x, foot_y, _ = self._get_feet_axes()
        foot_x = foot_x[..., :2]
        foot_y = foot_y[..., :2]
        foot_x = F.normalize(foot_x, dim=-1, eps=1.0e-8)
        foot_y = F.normalize(foot_y, dim=-1, eps=1.0e-8)
        return foot_x, foot_y

    def _signed_margin_to_local_rect(self, point_xy, foot_center_xy, foot_x, foot_y, min_x, max_x, min_y, max_y):
        rel = point_xy - foot_center_xy
        local_x = torch.sum(rel * foot_x, dim=-1)
        local_y = torch.sum(rel * foot_y, dim=-1)

        clamped_x = torch.clamp(local_x, min=min_x, max=max_x)
        clamped_y = torch.clamp(local_y, min=min_y, max=max_y)
        outside_dist = torch.sqrt((local_x - clamped_x) ** 2 + (local_y - clamped_y) ** 2)

        inside_margin = torch.minimum(
            torch.minimum(max_x - local_x, local_x - min_x),
            torch.minimum(max_y - local_y, local_y - min_y),
        )
        inside_mask = (
            (local_x >= min_x)
            & (local_x <= max_x)
            & (local_y >= min_y)
            & (local_y <= max_y)
        )
        return torch.where(inside_mask, inside_margin, -outside_dist)

    def _signed_margin_to_foot_rect(self, point_xy, foot_center_xy, foot_x, foot_y, front, back, left, right):
        min_x = torch.full_like(point_xy[:, 0], -back)
        max_x = torch.full_like(point_xy[:, 0], front)
        min_y = torch.full_like(point_xy[:, 0], -right)
        max_y = torch.full_like(point_xy[:, 0], left)
        return self._signed_margin_to_local_rect(point_xy, foot_center_xy, foot_x, foot_y, min_x, max_x, min_y, max_y)

    def _build_foot_corners_world(self, foot_center, foot_x, foot_y, foot_z, front, back, left, right, sole_z):
        front_left = foot_center + front * foot_x + left * foot_y + sole_z * foot_z
        front_right = foot_center + front * foot_x - right * foot_y + sole_z * foot_z
        back_right = foot_center - back * foot_x - right * foot_y + sole_z * foot_z
        back_left = foot_center - back * foot_x + left * foot_y + sole_z * foot_z
        stack_dim = max(front_left.dim() - 1, 0)
        return torch.stack((front_left, front_right, back_right, back_left), dim=stack_dim)

    def _point_to_segment_distance(self, point_xy, seg_start, seg_end):
        seg_vec = seg_end - seg_start
        seg_len_sq = torch.sum(seg_vec ** 2, dim=-1).clamp_min(1.0e-8)
        rel = point_xy - seg_start
        t = torch.sum(rel * seg_vec, dim=-1) / seg_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        projections = seg_start + t.unsqueeze(-1) * seg_vec
        return torch.norm(point_xy - projections, dim=-1)

    def _get_active_support_geometry(self):
        front, back, left, right, sole_z, corner_tol = self._get_support_polygon_params()
        foot_centers = self.rigid_body_states[:, self.feet_indices, :3]
        foot_x, foot_y, foot_z = self._get_feet_axes()
        foot_corners_world = self._build_foot_corners_world(
            foot_centers,
            foot_x,
            foot_y,
            foot_z,
            front,
            back,
            left,
            right,
            sole_z,
        )

        contact_status = self.contact_filt
        foot_corner_z = foot_corners_world[..., 2]
        foot_min_z = foot_corner_z.min(dim=-1, keepdim=True).values
        active_corner_mask = contact_status.unsqueeze(-1) & (foot_corner_z <= foot_min_z + corner_tol)

        active_corner_count = active_corner_mask.sum(dim=(1, 2))
        active_corner_count_clamped = active_corner_count.clamp_min(1).to(foot_corner_z.dtype)
        support_height = torch.sum(
            foot_corner_z * active_corner_mask.to(foot_corner_z.dtype),
            dim=(1, 2),
        ) / active_corner_count_clamped
        support_height[active_corner_count == 0] = 0.0

        return (
            foot_corners_world,
            active_corner_mask,
            support_height,
        )

    def _batched_convex_hull_edges(self, points, point_mask=None):
        batch_size, num_points, _ = points.shape
        batch_idx = torch.arange(batch_size, device=points.device)
        if point_mask is None:
            point_mask = torch.ones(batch_size, num_points, dtype=torch.bool, device=points.device)

        has_points = point_mask.any(dim=1)

        inf = torch.full((batch_size, num_points), float("inf"), device=points.device, dtype=points.dtype)
        scores = torch.where(point_mask, points[..., 0] * 1000.0 + points[..., 1], inf)
        start = torch.argmin(scores, dim=1)
        start = torch.where(has_points, start, torch.zeros_like(start))
        hull_idx = torch.zeros(batch_size, num_points + 1, dtype=torch.long, device=points.device)
        hull_idx[:, 0] = start

        current = start.clone()
        active = has_points.clone()
        eps = 1.0e-8

        for k in range(1, num_points + 1):
            candidate = current.clone()
            candidate_found = torch.zeros(batch_size, dtype=torch.bool, device=points.device)
            for offset in range(1, num_points + 1):
                candidate_try = (current + offset) % num_points
                valid_try = (
                    active
                    & (~candidate_found)
                    & point_mask[batch_idx, candidate_try]
                    & (candidate_try != current)
                )
                candidate = torch.where(valid_try, candidate_try, candidate)
                candidate_found |= valid_try

            step_active = active & candidate_found

            current_points = points[batch_idx, current]
            for j in range(num_points):
                j_idx = torch.full_like(candidate, j)
                same_current = j_idx == current

                candidate_points = points[batch_idx, candidate]
                point_j = points[:, j, :]

                edge_candidate = candidate_points - current_points
                edge_j = point_j - current_points
                cross = edge_candidate[:, 0] * edge_j[:, 1] - edge_candidate[:, 1] * edge_j[:, 0]
                dist_candidate = torch.sum(edge_candidate ** 2, dim=-1)
                dist_j = torch.sum(edge_j ** 2, dim=-1)

                better = step_active & point_mask[:, j] & (~same_current) & (
                    (cross < -eps) | ((torch.abs(cross) <= eps) & (dist_j > dist_candidate))
                )
                candidate = torch.where(better, j_idx, candidate)

            candidate = torch.where(step_active, candidate, current)
            hull_idx[:, k] = candidate
            returned_to_start = step_active & (candidate == start)
            current = torch.where(step_active, candidate, current)
            active = step_active & (~returned_to_start)

        edge_start_idx = hull_idx[:, :-1]
        edge_end_idx = hull_idx[:, 1:]
        edge_mask = edge_end_idx != edge_start_idx
        edge_starts = points[batch_idx[:, None], edge_start_idx]
        edge_ends = points[batch_idx[:, None], edge_end_idx]
        return edge_starts, edge_ends, edge_mask

    def _signed_margin_to_convex_polygon(self, point_xy, edge_starts, edge_ends, edge_mask):
        edge_vec = edge_ends - edge_starts
        edge_len = torch.norm(edge_vec, dim=-1).clamp_min(1.0e-8)
        edge_unit = edge_vec / edge_len.unsqueeze(-1)
        rel = point_xy[:, None, :] - edge_starts

        signed_line_dist = edge_unit[..., 0] * rel[..., 1] - edge_unit[..., 1] * rel[..., 0]
        inf = torch.full_like(signed_line_dist, float("inf"))
        signed_line_dist = torch.where(edge_mask, signed_line_dist, inf)
        inside_margin = signed_line_dist.min(dim=1).values

        edge_len_sq = torch.sum(edge_vec ** 2, dim=-1).clamp_min(1.0e-8)
        t = torch.sum(rel * edge_vec, dim=-1) / edge_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        projections = edge_starts + t.unsqueeze(-1) * edge_vec
        seg_dist = torch.norm(point_xy[:, None, :] - projections, dim=-1)
        seg_dist = torch.where(edge_mask, seg_dist, inf)
        outside_dist = seg_dist.min(dim=1).values

        inside_mask = inside_margin >= 0.0
        return torch.where(inside_mask, inside_margin, -outside_dist)

    def _signed_margin_to_support_patch(self, point_xy, support_points, support_mask):
        edge_starts, edge_ends, edge_mask = self._batched_convex_hull_edges(support_points, support_mask)
        edge_counts = edge_mask.sum(dim=1)
        margin = torch.zeros(point_xy.shape[0], device=point_xy.device, dtype=point_xy.dtype)

        polygon_mask = edge_counts >= 3
        if polygon_mask.any():
            margin[polygon_mask] = self._signed_margin_to_convex_polygon(
                point_xy[polygon_mask],
                edge_starts[polygon_mask],
                edge_ends[polygon_mask],
                edge_mask[polygon_mask],
            )

        line_mask = edge_counts == 2
        if line_mask.any():
            first_edge_idx = edge_mask[line_mask].to(torch.int64).argmax(dim=1)
            batch_idx = torch.arange(first_edge_idx.shape[0], device=point_xy.device)
            seg_start = edge_starts[line_mask][batch_idx, first_edge_idx]
            seg_end = edge_ends[line_mask][batch_idx, first_edge_idx]
            margin[line_mask] = -self._point_to_segment_distance(
                point_xy[line_mask],
                seg_start,
                seg_end,
            )

        point_mask = edge_counts <= 1
        if point_mask.any():
            first_point_idx = support_mask[point_mask].to(torch.int64).argmax(dim=1)
            batch_idx = torch.arange(first_point_idx.shape[0], device=point_xy.device)
            support_point = support_points[point_mask][batch_idx, first_point_idx]
            margin[point_mask] = -torch.norm(point_xy[point_mask] - support_point, dim=-1)

        return margin

    def compute_zmp(self, update_history=True):
        total_mass = torch.zeros(self.num_envs, device=self.device)
        self.weighted_position_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.weighted_velocity_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        for i in range(22):
            mass = self.mass_tensor[:, i]
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
        (
            foot_corners_world,
            active_corner_mask,
            support_height,
        ) = self._get_active_support_geometry()
        self.support_corners_world.copy_(foot_corners_world)
        self.active_support_corner_mask.copy_(active_corner_mask)
        self.support_plane_height.copy_(support_height)

        self.com_pos[:, 2] = self.com_pos[:, 2] - support_height

        # Step 1: Calculate ZMP position for each environment. Recovery can make
        # one-step COM acceleration estimates very noisy, so smooth COM velocity
        # before differentiating it.
        filter_alpha = getattr(self.cfg.rewards, "zmp_com_vel_filter_alpha", 1.0)
        filtered_com_vel = (1.0 - filter_alpha) * self.filtered_com_vel + filter_alpha * com_vel
        com_acc = (filtered_com_vel - self.last_filtered_com_vel) / self.dt

        self.zmp_x = self.com_pos[:,0] - (self.com_pos[:,2] / 9.81) * com_acc[:, 0]
        self.zmp_y = self.com_pos[:,1] - (self.com_pos[:,2] / 9.81) * com_acc[:, 1]

        if update_history:
            self.last_com_vel.copy_(com_vel)
            self.last_filtered_com_vel.copy_(filtered_com_vel)
            self.filtered_com_vel.copy_(filtered_com_vel)

        has_contact_mask = contact_status[:, 0] | contact_status[:, 1]
        left_support_mask = contact_status[:, 0] & (~contact_status[:, 1])
        right_support_mask = (~contact_status[:, 0]) & contact_status[:, 1]
        double_support_mask = contact_status[:, 0] & contact_status[:, 1]

        self.zmp_xy = torch.stack((self.zmp_x, self.zmp_y), dim=-1)
        self.zmp_distance.zero_()
        self.stability_margin.zero_()
        self.zmp_valid_mask[:] = has_contact_mask

        if left_support_mask.any():
            margin = self._signed_margin_to_support_patch(
                self.zmp_xy[left_support_mask],
                foot_corners_world[left_support_mask, 0, :, :2],
                active_corner_mask[left_support_mask, 0],
            )
            self.stability_margin[left_support_mask, 0] = margin

        if right_support_mask.any():
            margin = self._signed_margin_to_support_patch(
                self.zmp_xy[right_support_mask],
                foot_corners_world[right_support_mask, 1, :, :2],
                active_corner_mask[right_support_mask, 1],
            )
            self.stability_margin[right_support_mask, 0] = margin

        if double_support_mask.any():
            polygon_points = torch.cat(
                (
                    foot_corners_world[double_support_mask, 0, :, :2],
                    foot_corners_world[double_support_mask, 1, :, :2],
                ),
                dim=1,
            )
            polygon_mask = torch.cat(
                (
                    active_corner_mask[double_support_mask, 0],
                    active_corner_mask[double_support_mask, 1],
                ),
                dim=1,
            )
            margin = self._signed_margin_to_support_patch(
                self.zmp_xy[double_support_mask],
                polygon_points,
                polygon_mask,
            )
            self.stability_margin[double_support_mask, 0] = margin

        self.zmp_distance[has_contact_mask, 0] = torch.relu(-self.stability_margin[has_contact_mask, 0])

    def _compute_zmp_cost(self):
        self.cost_buf.zero_()
        if not getattr(self.cfg.rewards, "use_zmp_cost", False):
            return self.cost_buf

        zmp_cost_type = getattr(self.cfg.rewards, "zmp_cost_type", "indicator")
        valid_mask = self.zmp_valid_mask
        if zmp_cost_type == "indicator":
            self.cost_buf[valid_mask] = (self.stability_margin[valid_mask, 0] < 0.0).float()
        elif zmp_cost_type == "margin":
            slack = getattr(self.cfg.rewards, "zmp_margin_slack", 0.0)
            outside = torch.relu(-self.stability_margin[valid_mask, 0] - slack)
            clip = getattr(self.cfg.rewards, "zmp_cost_clip", 0.0)
            if clip > 0.0:
                outside = torch.clamp(outside, max=clip)

            support_weight = torch.ones_like(outside)
            contact_status = self.contact_filt[valid_mask]
            double_support = contact_status[:, 0] & contact_status[:, 1]
            single_support = contact_status[:, 0] ^ contact_status[:, 1]
            support_weight[double_support] = getattr(self.cfg.rewards, "zmp_double_support_weight", 1.0)
            support_weight[single_support] = getattr(self.cfg.rewards, "zmp_single_support_weight", 1.0)

            self.cost_buf[valid_mask] = outside * support_weight
        else:
            raise ValueError(f"Unsupported zmp_cost_type: {zmp_cost_type}")
        if (~valid_mask).any():
            self.cost_buf[~valid_mask] = getattr(self.cfg.rewards, "zmp_no_contact_cost", 0.0)
        return self.cost_buf

    
    def step(self, actions):
        actions = self.reindex(actions).to(self.device)
        actions.to(self.device)
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms


        self.global_counter += 1
        self.total_env_steps_counter += 1
        
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
                
        
        self.actions[:, [4, 9]] = torch.clamp(self.actions[:, [4, 9]], -1.0, 1.0)

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()


        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

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
        batch_size = len(env_ids)
        indices = np.random.choice(len(self.extreme_data), batch_size, replace=False)
        sampled_data = torch.tensor([self.extreme_data[i] for i in indices]).to(self.device)
        self.root_states[env_ids, 7:10] = sampled_data[:, :3]
        self.root_states[env_ids, 10:13] = sampled_data[:, 3:6]
        rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        rand_roll = sampled_data[:, 6]
        rand_pitch = sampled_data[:, 7]
        quat = quat_from_euler_xyz(rand_roll, rand_pitch, rand_yaw) 
        self.root_states[env_ids, 3:7] = quat[:, :] 
        self.dof_pos[env_ids] = sampled_data[:, 8:27]
        self.dof_vel[env_ids] = sampled_data[:, 27:46]
        self.commands[env_ids, :3] = sampled_data[:, 46:49]
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
            if flag > 0.01:   # 0.1   0.05
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
        self.last_com_vel[env_ids] = 0.
        self.filtered_com_vel[env_ids] = 0.
        self.last_filtered_com_vel[env_ids] = 0.
        self.zmp_distance[env_ids] = 0.
        self.stability_margin[env_ids] = 0.
        self.zmp_valid_mask[env_ids] = False
        self.cost_buf[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset observation history buffer
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if getattr(self.cfg.rewards, "use_zmp_cost", False):
            self.extras["episode"]["cost_zmp"] = torch.mean(self.episode_cost_sums[env_ids]) / self.max_episode_length_s
            self.episode_cost_sums[env_ids] = 0.
        self.extras["episode"]["reset_contact"] = torch.mean(self.reset_reason_contact[env_ids])
        self.extras["episode"]["reset_contact_pelvis"] = torch.mean(self.reset_reason_pelvis_contact[env_ids])
        self.extras["episode"]["reset_contact_hip"] = torch.mean(self.reset_reason_hip_contact[env_ids])
        self.extras["episode"]["reset_contact_knee"] = torch.mean(self.reset_reason_knee_contact[env_ids])
        self.extras["episode"]["reset_roll"] = torch.mean(self.reset_reason_roll[env_ids])
        self.extras["episode"]["reset_pitch"] = torch.mean(self.reset_reason_pitch[env_ids])
        self.extras["episode"]["reset_timeout"] = torch.mean(self.reset_reason_timeout[env_ids])
        self.reset_reason_contact[env_ids] = 0.
        self.reset_reason_pelvis_contact[env_ids] = 0.
        self.reset_reason_hip_contact[env_ids] = 0.
        self.reset_reason_knee_contact[env_ids] = 0.
        self.reset_reason_roll[env_ids] = 0.
        self.reset_reason_pitch[env_ids] = 0.
        self.reset_reason_timeout[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0

        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["command_curriculum"] = self.command_curriculum_progress
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return

    def _resample_commands(self, env_ids):
        self._update_recovery_command_curriculum_ranges()
        super()._resample_commands(env_ids)

   
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
                self.root_states[env_ids, :2] += torch_rand_float(-2.0, 2.0, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
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
            if self.cfg.env.rand_vel:
                # pos
                self.root_states[env_ids, :1] += torch_rand_float(-1.0, 15.0, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
                self.root_states[env_ids, 1:2] += torch_rand_float(-2.0, 2.0, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
                
                terrain_height = self._get_heights()
                self.root_states[env_ids, 2] = 1.1
                self.root_states[env_ids, 2] += terrain_height[env_ids, 66]
                
                # vel
                self.root_states[env_ids, 7:8] = torch_rand_float(self.cfg.env.rand_lin_x_vel[0], self.cfg.env.rand_lin_x_vel[1], \
                                                                        (len(env_ids), 1), device=self.device)  # [7:10]: lin vel
                self.root_states[env_ids, 8:9] = torch_rand_float(self.cfg.env.rand_lin_y_vel[0], self.cfg.env.rand_lin_y_vel[1], \
                                                                        (len(env_ids), 1), device=self.device)  # [7:10]: lin vel
                self.root_states[env_ids, 9:10] = torch_rand_float(self.cfg.env.rand_lin_z_vel[0], self.cfg.env.rand_lin_z_vel[1], \
                                                                        (len(env_ids), 1), device=self.device)  # [7:10]: lin vel
                self.root_states[env_ids, 10:13] = torch_rand_float(self.cfg.env.rand_ang_vel[0], self.cfg.env.rand_ang_vel[1], \
                                                                        (len(env_ids), 3), device=self.device)  # [10:13]: ang vel
            self.initial_origins[env_ids, :3] = self.root_states[env_ids, :3]
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
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
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.2, 1.8, (len(env_ids), self.num_dofs), device=self.device)
            self.dof_vel[env_ids] = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dofs), device=self.device)
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
        return

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        self._random_force()

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

    def _random_force(self):
        # For Force
        force = torch.zeros(self.num_envs * len(self._body_list),  3,  device=self.device)
        max_force = self.cfg.domain_rand.max_force
        self.continuous_force[:, :] = torch_rand_float(-max_force, max_force, (len(self._body_list) * self.num_envs, 3), device=self.device)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.continuous_force), None, gymapi.LOCAL_SPACE)

        # For Torque
        max_torque = self.cfg.domain_rand.max_torque
        torque = torch.zeros(self.num_envs * len(self._body_list),  3,  device=self.device)
        self.continuous_torque[:, :] = torch_rand_float(-max_torque, max_torque, (len(self._body_list) * self.num_envs, 3), device=self.device)
        self.gym.apply_rigid_body_force_tensors(self.sim,  None, gymtorch.unwrap_tensor(self.continuous_torque), gymapi.LOCAL_SPACE)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force
        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)
        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_obs_buf_commands(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,
                            imu_obs,
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.last_actions,
                            self.projected_gravity
                            ),dim=-1)
    
    def frequency_encoding(self, zmp_feature, F):
        encoding = []
        for i in range(F):
            freq = 2 ** i
            encoding.append(torch.sin(freq * torch.pi * zmp_feature))
            encoding.append(torch.cos(freq * torch.pi * zmp_feature))
        return torch.cat(encoding, dim=-1)
    
    def compute_observations(self):
        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)#self._demo_obs_buf[:, 2:, :].clone().flatten(start_dim=1) 
        priv_motion_features = self.priv_obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)

        # Terrain height
        heights = torch.clip(self.measured_heights, -1., 1.)

        measured_heights = torch.sum(self.rigid_body_states[:, self.feet_indices, 2], dim=1) / 2.0
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)

        # If add ZMP encoding to the observation
        self.zmp = self.frequency_encoding(self.zmp_distance, 4)

        # # Only estimate base linear velocity
        state_labels = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.stability_margin,
                base_height.unsqueeze(1),
            ),
            dim=-1,
        )
        domain_rand_labels = torch.cat(
            (
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1,
                self.motor_strength[1] - 1,
                self._kp_scale,
                self._kd_scale,
                self.rand_push_force,
                self.rand_push_torque,
            ),
            dim=-1,
        )
        estimator_labels = torch.cat((state_labels, domain_rand_labels), dim=-1)

        # Privileged information for Critic
        priv_latent = torch.cat(
            (
                domain_rand_labels,
                self.continuous_force.reshape(self.num_envs, -1),
                self.continuous_torque.reshape(self.num_envs, -1),
                heights,
            ),
            dim=-1,
        )

        obs_buf = self.compute_obs_buf_commands()

        # Gait reference commands
        if self.cfg.task.use_gait:
            phase = self._get_phase()
            sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
            cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
            self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3]), dim=1)
        else:
            self.command_input = self.commands[:, :3]
        
        obs_buf = torch.cat((obs_buf, self.command_input), dim = -1)

        if self.cfg.env.extreme_flag == True:
            flag = np.random.rand()
            if flag > 0.98:
                obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale * 3
            else:
                obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale * 1.5
        else:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale

        if self.train_estimator == True:
            self.obs_buf = torch.cat([motion_features, obs_buf, estimator_labels], dim=-1)
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf], dim=-1)

        priv_obs_buf = torch.cat([obs_buf, priv_latent, estimator_labels], dim=-1)
        self.privileged_obs_buf = torch.cat([priv_motion_features, priv_obs_buf], dim=-1)

        if self.cfg.env.history_len != 0:
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

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_mask = torch.norm(
            self.contact_forces[:, self.termination_contact_indices, :], dim=-1
        ) > 1.0
        contact_reset = torch.any(termination_contact_mask, dim=1)
        roll_cutoff = torch.abs(self.roll) > 1.0
        pitch_cutoff = torch.abs(self.pitch) > 1.0
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_reason_contact[:] = contact_reset.float()
        self.reset_reason_pelvis_contact[:] = self._reset_reason_from_contact_group(termination_contact_mask, "pelvis")
        self.reset_reason_hip_contact[:] = self._reset_reason_from_contact_group(termination_contact_mask, "hip")
        self.reset_reason_knee_contact[:] = self._reset_reason_from_contact_group(termination_contact_mask, "knee")
        self.reset_reason_roll[:] = roll_cutoff.float()
        self.reset_reason_pitch[:] = pitch_cutoff.float()
        self.reset_reason_timeout[:] = self.time_out_buf.float()
        self.reset_buf = contact_reset
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff

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
        # Let the robot stand still when command is 0
        mask = torch.norm(self.commands[:, :2], dim=1) < 0.1
        stance_mask[mask] = 1
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
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_action = 2 * self.ref_dof_pos

    # ######### Rewards #########
    def compute_reward(self):
        # Refresh ZMP-related signals before reward/cost computation so recovery cost
        # uses the current post-step state rather than the previous observation update.
        self.compute_zmp(update_history=True)
        self._compute_zmp_cost()
        self.extras["zmp_cost"] = self.cost_buf.clone()
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
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

        if getattr(self.cfg.rewards, "use_zmp_cost", False):
            self.episode_cost_sums += self.cost_buf

    def _reward_lin_vel_z(self):
        rew = torch.square(self.base_lin_vel[:, 2])
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

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
        joint_diff = self.dof_pos[:, :11] - self.default_dof_pos[:, :11]
        torso_diff = self.dof_pos[:, 10] - self.default_dof_pos[:, 10]
        return - 0.1 * torch.norm(joint_diff, dim=1)
    
    def _reward_upper_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        shoulder_pitch_diff = torch.abs(self.dof_pos[:, 11] - self.default_dof_pos[:, 11]) + torch.abs(self.dof_pos[:, 15] - self.default_dof_pos[:, 15])
        shoulder_roll_diff = torch.abs(self.dof_pos[:, 12]  - self.default_dof_pos[:, 12]) + torch.abs(self.dof_pos[:, 16] - self.default_dof_pos[:, 16])
        shoulder_yaw_diff = torch.abs(self.dof_pos[:, 13] - self.default_dof_pos[:, 13]) + torch.abs(self.dof_pos[:, 17] - self.default_dof_pos[:, 17])
        torso_diff = self.dof_pos[:, 10] - self.default_dof_pos[:, 10]
        elbow_left_diff = self.dof_pos[:, 14] - self.default_dof_pos[:, 14]
        elbow_right_diff = self.dof_pos[:, 18] - self.default_dof_pos[:, 18]
        return - 0.1 * torch.abs(torso_diff) - 0.1 * torch.abs(shoulder_roll_diff) - 0.1 * torch.abs(shoulder_yaw_diff) - 0.05 * torch.abs(elbow_left_diff) - 0.05 * torch.abs(elbow_right_diff) - 0.08 * torch.abs(shoulder_pitch_diff)
        
    def _reward_zmp_alignment(self):
        """
        Encourages the projection of the ZMP to be close to the support center.
        Args:
            pcsp_xy: (N, 2) support center positions on the ground plane
            zmp_xy: (N, 2) ZMP projection on the ground plane
            scale: controls the width of the reward
        Returns:
            (N,) tensor with ZMP alignment reward
        """
        dist_sq = torch.sum((pcsp_xy - zmp_xy) ** 2, dim=-1)
        return torch.exp(-dist_sq / scale)

    def _reward_angular_momentum(self):
        pos = self.rigid_body_states[..., 0:3]         # [E, B, 3]
        lin_vel = self.rigid_body_states[..., 7:10]    # [E, B, 3]
        ang_vel = self.rigid_body_states[..., 10:13]   # [E, B, 3]

        mass = self.mass_tensor                        # [E, B]
        inertia = self.inertia_tensor                  # [E, B, 3, 3]

        mv = mass.unsqueeze(-1) * lin_vel              # [E, B, 3]
        p_cross_mv = torch.cross(pos, mv, dim=-1)      # [E, B, 3]
        Iw = torch.einsum("ebij,ebj->ebi", inertia, ang_vel)  # [E, B, 3]

        L_total = torch.sum(p_cross_mv + Iw, dim=1)    # [E, 3]
        L_norm = torch.norm(L_total, dim=-1)           # [E]
        reward = torch.exp(-L_norm / 5.0)              # [E]
        return reward

    def _reward_lower_stand(self):
        # Penalize deviation from the nominal standing posture.
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        measured_heights = torch.sum(
            self.rigid_body_states[:, self.feet_indices, 2], dim=1) / 2.0
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 20) # 50

    def _reward_zmp_distance(self):
        zmp_rew = 1.0 / (self.zmp_distance.squeeze() + 1.0)
        mask = self.zmp_distance.squeeze() < 0.2
        zmp_rew[mask] = 0.
        return zmp_rew

    def _reward_zmp_distance_exp(self):
        reward = torch.exp(-self.zmp_distance.squeeze() / 0.05) 
        return reward

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


    def _reward_low_stand(self):
        # Penalize motion at zero commands
        low_stand_pos = self.default_dof_pos.clone()
        low_stand_pos[:, [2, 7]] += -0.2  
        low_stand_pos[:, [3, 8]] += 0.2

        return torch.sum(torch.abs(self.dof_pos - low_stand_pos), dim=1)

    def _reward_feet_drag(self):
        # feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
        feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:9]).sum(dim=-1)
        dragging_vel = self.contact_filt * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)

        rew = torch.mean(reward, dim=1)
        return rew

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
        
        mask = torch.norm(self.commands[:, :2], dim=1) < 0.1
        rew_pos[mask] = 0.0
        return rew_pos


    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos[:, :10].clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        mask = torch.norm(self.commands[:, :2], dim=1) < 0.1
        r[mask] = 0.0
        return r

    def _reward_feet_height(self):
        terrain_height = self._get_heights()
        feet_height = self.rigid_body_states[:, self.feet_indices, 2] - terrain_height[:, 66].unsqueeze(1)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        height_diff = torch.square(feet_height - 0.1) * ~ contact
        rew = torch.sum(height_diff, dim=(1))

        mask = torch.norm(self.commands[:, :2], dim=1) < 0.1
        rew[mask] = 0.0
        return rew    

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
