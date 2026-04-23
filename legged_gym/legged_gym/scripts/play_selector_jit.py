import argparse
import os
from pathlib import Path

_MOTION_TASK_ENV_VAR = "LEGGED_GYM_MOTION_TASK"


def _apply_motion_task_cli_override():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion_task", choices=("walk", "recovery"))
    early_args, _ = parser.parse_known_args()
    if early_args.motion_task is not None:
        os.environ[_MOTION_TASK_ENV_VAR] = early_args.motion_task


_apply_motion_task_cli_override()

import imageio.v2 as imageio
import isaacgym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _extract_estimator_latent(estimator_output):
    primary = estimator_output[0] if isinstance(estimator_output, (tuple, list)) else estimator_output
    if not isinstance(primary, (tuple, list)) or len(primary) < 2:
        raise TypeError(f"Unexpected estimator output structure: {type(estimator_output)}")

    z, v = primary[0], primary[1]
    if not isinstance(z, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError(f"Estimator latent outputs must be tensors, got {type(z)} and {type(v)}")
    return z, v


class SelectorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=2, dropout_prob=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.dropout = nn.Identity() if dropout_prob <= 0.0 else nn.Dropout(p=dropout_prob)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def _require_path(path_value, arg_name):
    if not path_value:
        raise ValueError(f"Please provide {arg_name}.")
    path = os.path.abspath(path_value)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{arg_name} not found: {path}")
    return path


def _compute_latent(policy, env_obs, priv_start):
    z, v = _extract_estimator_latent(policy.estimator(env_obs.detach()[:, :priv_start]))
    return torch.cat([z, v], dim=1)


def _build_low_level_obs(env_obs, latent, prop_start, prop_dim):
    obs_end = prop_start + prop_dim
    return torch.cat([env_obs[:, :obs_end], latent], dim=1)


def _build_selector_obs(low_level_obs, prev_action):
    return torch.cat([low_level_obs, prev_action.unsqueeze(1)], dim=1)


def _prepare_video_writer(args, selector_path):
    if not args.record_video:
        return None, None

    video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "videos_selector")
    os.makedirs(video_dir, exist_ok=True)
    selector_name = Path(selector_path).stem
    motion_task = args.motion_task if args.motion_task is not None else "walk"
    video_path = os.path.join(video_dir, f"{selector_name}_{motion_task}.mp4")
    writer = imageio.get_writer(video_path, fps=25)
    return writer, video_path


def play(args):
    if args.task != "h1_selector":
        print(f"Using task '{args.task}'. For H1 high-level evaluation the recommended task is 'h1_selector'.")

    selector_path = _require_path(args.selector_path, "--selector_path")
    loco_path = _require_path(args.loco_jit, "--loco_jit")
    reco_path = _require_path(args.reco_jit, "--reco_jit")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.num_envs is None:
        env_cfg.env.num_envs = 1 if args.record_video else 128
    if args.record_video:
        env_cfg.env.record_video = True
        env_cfg.env.record_frame = False
    env_cfg.terrain.curriculum = False

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.enable_viewer_sync = not args.record_video

    selector_input = env_cfg.env.n_feature + env_cfg.env.n_proprio + env_cfg.env.n_decoder_out + 1
    selector_dropout = getattr(train_cfg.policy, "selector_dropout_prob", 0.0)
    selector = SelectorNetwork(selector_input, dropout_prob=selector_dropout).to(env.device)
    selector.load_state_dict(torch.load(selector_path, map_location=env.device))
    selector.eval()

    locomotion_policy = torch.jit.load(loco_path, map_location=env.device).to(env.device)
    recovery_policy = torch.jit.load(reco_path, map_location=env.device).to(env.device)

    print("selector checkpoint:", selector_path)
    print("locomotion jit:", loco_path)
    print("recovery jit:", reco_path)
    print("num_envs:", env.num_envs)

    writer, video_path = _prepare_video_writer(args, selector_path)

    obs = env.get_observations()
    latent = _compute_latent(locomotion_policy, obs, train_cfg.estimator.priv_start)
    low_level_obs = _build_low_level_obs(obs, latent, train_cfg.estimator.prop_start, train_cfg.estimator.prop_dim)
    prev_action = torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)
    selector_obs = _build_selector_obs(low_level_obs, prev_action)

    total_reward = torch.zeros(env.num_envs, device=env.device)
    total_switches = torch.zeros(env.num_envs, device=env.device)
    locomotion_count = torch.zeros(env.num_envs, device=env.device)
    recovery_count = torch.zeros(env.num_envs, device=env.device)

    traj_length = int(env.max_episode_length)

    try:
        with torch.no_grad():
            for _ in tqdm(range(traj_length)):
                q_values = selector(selector_obs)
                selection = q_values.argmax(dim=1).long()
                selection_float = selection.float()

                has_prev_action = prev_action >= 0.0
                switched = has_prev_action & (prev_action != selection_float)
                total_switches += switched.float()

                locomotion_mask = selection == 0
                recovery_mask = selection == 1
                locomotion_count += locomotion_mask.float()
                recovery_count += recovery_mask.float()

                actions = torch.empty((env.num_envs, env.num_actions), device=env.device)
                if locomotion_mask.any():
                    actions[locomotion_mask] = locomotion_policy(low_level_obs[locomotion_mask])
                if recovery_mask.any():
                    actions[recovery_mask] = recovery_policy(low_level_obs[recovery_mask])

                next_obs, _, reward, dones, infos = env.step(actions)
                reward = reward.view(-1)
                dones = dones.view(-1)
                total_reward += reward

                if locomotion_mask.any():
                    loco_latent = _compute_latent(locomotion_policy, next_obs, train_cfg.estimator.priv_start)
                    latent[locomotion_mask] = loco_latent[locomotion_mask]
                if recovery_mask.any():
                    reco_latent = _compute_latent(recovery_policy, next_obs, train_cfg.estimator.priv_start)
                    latent[recovery_mask] = reco_latent[recovery_mask]

                low_level_obs = _build_low_level_obs(next_obs, latent, train_cfg.estimator.prop_start, train_cfg.estimator.prop_dim)
                prev_action = selection_float.clone()
                done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                if len(done_ids) > 0:
                    prev_action[done_ids] = -1.0
                selector_obs = _build_selector_obs(low_level_obs, prev_action)

                if writer is not None:
                    imgs = env.render_record(mode="rgb_array")
                    if imgs is not None and len(imgs) > 0:
                        writer.append_data(imgs[0][..., :3])
    finally:
        if writer is not None:
            writer.close()

    total_steps = locomotion_count + recovery_count
    locomotion_ratio = (locomotion_count / (total_steps + 1.0e-6)).mean().item()
    recovery_ratio = (recovery_count / (total_steps + 1.0e-6)).mean().item()
    mean_reward = total_reward.mean().item()
    mean_switches = total_switches.mean().item()

    print("Mean reward:", mean_reward)
    print("Locomotion ratio:", locomotion_ratio)
    print("Recovery ratio:", recovery_ratio)
    print("Mean switches per env:", mean_switches)
    if video_path is not None:
        print("Saved video to:", video_path)


if __name__ == "__main__":
    args = get_args()
    play(args)
