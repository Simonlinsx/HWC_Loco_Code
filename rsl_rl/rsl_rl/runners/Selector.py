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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime
import torch
from collections import deque
import random
import numpy as np


from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
from rsl_rl.utils.utils import Normalizer

import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import RolloutStorage, ReplayBuffer
import torch.nn.functional as F


def _extract_estimator_latent(estimator_output):
    if isinstance(estimator_output, (tuple, list)):
        primary = estimator_output[0]
    else:
        primary = estimator_output

    if not isinstance(primary, (tuple, list)) or len(primary) < 2:
        raise TypeError(
            f"Unexpected estimator output structure: {type(estimator_output)}"
        )

    z, v = primary[0], primary[1]
    if not isinstance(z, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError(
            f"Estimator latent outputs must be tensors, got {type(z)} and {type(v)}"
        )
    return z, v



# Replay Buffer
class ReplayBuffer_selector:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    # def add(self, state, action, reward, next_state, done):
        # self.buffer.append((state, action, reward, next_state, done))
    
    def add(self, state, action, reward, next_state, done):
        # self.buffer.append((
        #     np.array(state), 
        #     np.array(action), 
        #     np.array(reward), 
        #     np.array(next_state), 
        #     np.array(done)
        # ))
        self.buffer.append((
            np.asarray(state), 
            np.asarray(action), 
            np.asarray(reward), 
            np.asarray(next_state), 
            np.asarray(done)
        ))

    def sample(self, batch_size):
        # 从buffer中随机采样
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled_data = [self.buffer[i] for i in indices]

        # 解包数据
        # states, actions, rewards, next_states, dones = zip(*sampled_data)
        
        # 转换为 NumPy 数组
        states, actions, rewards, next_states, dones = map(np.array, zip(*sampled_data))


        # 转换为张量，确保形状正确
        return (
            torch.tensor(states).to(self.device),  # 确保为二维数组
            torch.tensor(actions).to(self.device),             # 动作为一维数组
            torch.tensor(rewards).to(self.device),          # 奖励为一维数组
            torch.tensor(next_states).to(self.device),  # 确保为二维数组
            torch.tensor(dones).to(self.device),            # 完成标志为一维数组
        )


    def __len__(self):
        return len(self.buffer)


# Reward Normalizer
class RewardNormalizer:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.running_mean = 0
        self.var = 1
        self.count = 1e-4

    def normalize(self, reward):
        self.running_mean = self.gamma * self.running_mean + (1 - self.gamma) * reward
        self.var = self.gamma * self.var + (1 - self.gamma) * (reward - self.running_mean) ** 2
        self.count += 1
        std = (self.var / self.count) ** 0.5
        return (reward - self.running_mean) / (std + 1e-8)


def _resolve_selector_paths(args, log_dir):
    loco_path = getattr(args, "loco_jit", None) if args is not None else None
    reco_path = getattr(args, "reco_jit", None) if args is not None else None

    if (loco_path is None) != (reco_path is None):
        raise ValueError("Please provide both --loco_jit and --reco_jit together for selector training.")

    if loco_path is None and reco_path is None:
        raise ValueError("Please provide --loco_jit and --reco_jit for selector training.")

    loco_path = os.path.abspath(loco_path)
    reco_path = os.path.abspath(reco_path)
    model_save_path = os.path.abspath(log_dir) if log_dir is not None else os.path.dirname(loco_path)
    os.makedirs(model_save_path, exist_ok=True)

    if not os.path.isfile(loco_path):
        raise FileNotFoundError(f"Selector locomotion JIT not found: {loco_path}")
    if not os.path.isfile(reco_path):
        raise FileNotFoundError(f"Selector recovery JIT not found: {reco_path}")

    return loco_path, reco_path, model_save_path

class SelectorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=2, dropout_prob=0.1):
        super(SelectorNetwork, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)  # 输出 Q 值，每个动作一个
        
        self.dropout = nn.Identity() if dropout_prob <= 0.0 else nn.Dropout(p=dropout_prob)
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)


    def forward(self, x):
        # 前向传播
        x = F.relu(self.fc1(x))  # 第一层，带 LayerNorm 和 ReLU
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # 第二层，带 LayerNorm 和 ReLU
        x = F.relu(self.fc3(x))  # 第三层，带 LayerNorm 和 ReLU
        x = self.fc4(x)                    # 输出层，不加激活函数
        return x


# RL Trainer
class Selector_Trainer:
    def __init__(self,
            env: VecEnv,
            train_cfg,
            log_dir=None,
            init_wandb=True,
            device='cpu', **kwargs):

        self.env = env
        self.device = env.device
        self.runner_cfg = train_cfg.get("runner", {})
        self.batch_size = self.runner_cfg.get("selector_batch_size", 512)
        self.lr = 1e-4
        self.gamma = 0.99
        self.max_grad_norm = self.runner_cfg.get("selector_max_grad_norm", 1.0)
        self.replay_buffer = ReplayBuffer_selector(
            capacity=self.runner_cfg.get("selector_replay_capacity", 2000000),
            device=self.device,
        )
        self.reward_normalizer = RewardNormalizer(gamma=self.gamma)
        self.updates_per_iteration = self.runner_cfg.get("selector_updates_per_iteration", 32)
        self.total_updates = 0

        self.prev_action = None  # 用于记录上一次的选择
        self.switch_penalty = self.runner_cfg.get("selector_switch_penalty", 0.00002)

        # Selector network and optimizer
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]

        # obs_dim = self.policy_cfg['selector_input'] + 1
        obs_dim = self.policy_cfg['selector_input']
        dropout_prob = self.policy_cfg.get("selector_dropout_prob", 0.0)
        self.selector = SelectorNetwork(input_dim=obs_dim, dropout_prob=dropout_prob).to(self.device)
        self.target_selector = deepcopy(self.selector).to(self.device)      # 目标网络
        self.target_selector.eval()  # 目标网络只推理，不训练
        self.target_update_frequency = self.runner_cfg.get("selector_target_update_frequency", 100)

        self.optimizer = optim.Adam(self.selector.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()  
        
        # ε-greedy 参数
        self.epsilon = self.runner_cfg.get("selector_epsilon_start", 0.1)
        self.epsilon_min = self.runner_cfg.get("selector_epsilon_min", 0.0002)
        self.epsilon_decay = self.runner_cfg.get("selector_epsilon_decay", 0.998)

        args = kwargs.get("args")
        loco_path, reco_path, self.model_save_path = _resolve_selector_paths(
            args=args,
            log_dir=log_dir,
        )
        print("current device is:", self.device)
        print("selector locomotion jit:", loco_path)
        print("selector recovery jit:", reco_path)
        print("selector save dir:", self.model_save_path)

        self.locomotion_policy = torch.jit.load(loco_path, map_location=self.device).to(self.device)
        self.recovery_policy =  torch.jit.load(reco_path, map_location=self.device).to(self.device)

        print("policy", self.locomotion_policy )


# Double-DQN
    def _compute_latent(self, policy, env_obs):
        z, v = _extract_estimator_latent(
            policy.estimator(env_obs.detach()[:, : self.estimator_cfg['priv_start']])
        )
        return torch.cat([z, v], dim=1)

    def _build_low_level_obs(self, env_obs, latent):
        obs_end = self.estimator_cfg['prop_start'] + self.estimator_cfg['prop_dim']
        return torch.cat([env_obs[:, :obs_end], latent], dim=1)

    def _build_selector_obs(self, low_level_obs, prev_action):
        return torch.cat([low_level_obs, prev_action.unsqueeze(1)], dim=1)

    def update_selector(self, update_step):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # print('states', states.shape)
        # Compute Q(s, a) using main network
        q_values = self.selector(states)  # 主网络输出 Q 值
        q_values = q_values.gather(1, actions.unsqueeze(-1).long()).squeeze(-1)  # 提取选择动作的 Q 值

        # Compute Q_target using target network
        with torch.no_grad():
            # 主网络选择下一状态的动作
            next_q_values_online = self.selector(next_states)
            max_action_indices = next_q_values_online.argmax(dim=1)
            # 目标网络评估所选动作的Q值
            next_q_values_target = self.target_selector(next_states)
            q_targets = rewards + self.gamma * next_q_values_target.gather(1, max_action_indices.unsqueeze(-1)).squeeze(-1) * (1 - dones.float())

        # Compute loss
        loss = self.loss_fn(q_values, q_targets)

        # Backward pass and optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.selector.parameters(), self.max_grad_norm)  # 梯度裁剪
        self.optimizer.step()

        # Update target network every few steps
        if update_step % self.target_update_frequency == 0:
            self.target_selector.load_state_dict(self.selector.state_dict())  # 同步参数

        return loss

    def select_action(self, q_values):
        # ε-greedy 策略
        if random.random() < self.epsilon:
            # print(torch.randint(0, 2, (q_values.shape[0]), device=self.device).long().shape)
            # print('random', torch.randint(0, 2, (q_values.shape[0],), device=self.device).long())
            return torch.randint(0, 2, (q_values.shape[0],), device=self.device).long()
        else:
            # print(q_values.argmax(dim=1).long().shape)
            return q_values.argmax(dim=1).long()  # 选择具有最大 Q 值的动作


    def save_selector(self, filename):
        """将 Selector Network 保存为 state_dict 格式"""
        # 获取模型的状态字典
        model_state_dict = self.selector.state_dict()
        
        # 保存为普通的字典格式
        filename = os.path.join(self.model_save_path, filename)
        torch.save(model_state_dict, filename)


    def load_selector(self, filename):
        path = os.path.join(self.model_save_path, filename)
        state_dict = torch.load(path, map_location=self.device)
        self.selector.load_state_dict(state_dict)
        self.selector.to(self.device)



    def learn(self, num_learning_iterations, num_steps_per_env, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_switch_penalty_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        switch_penalty_buf = deque(maxlen=100)
        
        # selected_action_history = torch.zeros(self.env.num_envs, 10 , device=self.device)

        frequence_period = 100

        locomotion_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        recovery_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        env_obs = self.env.get_observations()
        print(env_obs.shape)
        latent = self._compute_latent(self.locomotion_policy, env_obs)
        low_level_obs = self._build_low_level_obs(env_obs, latent)
        self.prev_action = torch.full((self.env.num_envs,), -1.0, dtype=torch.float32, device=self.device)
        selector_obs = self._build_selector_obs(low_level_obs, self.prev_action)

        ep_infos = []

        for it in range(num_learning_iterations):
            ep_infos = []
            with torch.inference_mode():
                for step in range(num_steps_per_env):
                    with torch.no_grad():
                        q_values = self.selector(selector_obs)
                        selected_action = self.select_action(q_values)  # 使用 ε-greedy 选择动作

                    selected_action_float = selected_action.float()
                    has_prev_action = self.prev_action >= 0.0
                    switch_penalty = -(
                        has_prev_action & (self.prev_action != selected_action_float)
                    ).float() * self.switch_penalty

                    # Execute the selected policy for all environments
                    actions = torch.empty((low_level_obs.shape[0], 19), device=self.device)  # Initialize action tensor

                    locomotion_mask = selected_action == 0  # Mask for locomotion policy
                    recovery_mask = selected_action == 1  # Mask for recovery policy

                    # Compute actions for each policy
                    if locomotion_mask.any():
                        actions[locomotion_mask] = self.locomotion_policy(low_level_obs[locomotion_mask])
                        locomotion_sum[locomotion_mask] +=1
                    if recovery_mask.any():
                        actions[recovery_mask] = self.recovery_policy(low_level_obs[recovery_mask])
                        recovery_sum[recovery_mask] +=1

                    # Interact with the environment
                    next_env_obs, privileged_obs, reward, dones, info = self.env.step(actions.to(self.device))
                    reward = reward.view(-1)
                    dones = dones.view(-1)

                    # Compute estimation and latent
                    if locomotion_mask.any():
                        loco_latent = self._compute_latent(self.locomotion_policy, next_env_obs)
                        latent[locomotion_mask] = loco_latent[locomotion_mask]
                    if recovery_mask.any():
                        reco_latent = self._compute_latent(self.recovery_policy, next_env_obs)
                        latent[recovery_mask] = reco_latent[recovery_mask]

                    next_low_level_obs = self._build_low_level_obs(next_env_obs, latent)
                    next_prev_action = selected_action_float.clone()
                    done_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(done_env_ids) > 0:
                        next_prev_action[done_env_ids] = -1.0
                    next_selector_obs = self._build_selector_obs(next_low_level_obs, next_prev_action)

                    normalized_reward = reward + switch_penalty

                    if 'episode' in info:
                        ep_infos.append(info['episode'])

                    cur_reward_sum += normalized_reward
                    cur_switch_penalty_sum += switch_penalty
                    cur_episode_length +=1

                    rewbuffer.extend(cur_reward_sum[done_env_ids].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[done_env_ids].cpu().numpy().tolist())
                    switch_penalty_buf.extend(cur_switch_penalty_sum[done_env_ids].cpu().numpy().tolist())

                    cur_reward_sum[done_env_ids] = 0
                    cur_switch_penalty_sum[done_env_ids] = 0
                    cur_episode_length[done_env_ids] = 0
                    locomotion_sum[done_env_ids] = 0
                    recovery_sum[done_env_ids] = 0

                    for i in range(selector_obs.shape[0]):  # 遍历每个环境的状态
                        self.replay_buffer.add(
                            selector_obs[i].detach().cpu(),         # 单个环境的状态
                            selected_action[i].detach().cpu(),      # 对应的动作
                            normalized_reward[i].detach().cpu(),    # 对应的奖励
                            next_selector_obs[i].detach().cpu(),    # 下一状态
                            dones[i].detach().cpu()                 # 完成标志
                        )
                    
                    selector_obs = next_selector_obs
                    low_level_obs = next_low_level_obs
                    self.prev_action = next_prev_action


            loco_percentage = (locomotion_sum / (locomotion_sum + recovery_sum + 1e-6)).mean()
            reco_percentage = (recovery_sum / (locomotion_sum + recovery_sum + 1e-6)).mean()

            # Update selector
            dqn_losses = []
            for _ in range(self.updates_per_iteration):
                dqn_loss = self.update_selector(update_step=self.total_updates)
                self.total_updates += 1
                if dqn_loss is not None:
                    dqn_losses.append(dqn_loss.item())
            mean_dqn_loss = float(np.mean(dqn_losses)) if len(dqn_losses) > 0 else 0.0

            # 衰减 ε
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


            # Logging
            # 打印本次迭代的 locomotion 策略选择的平均百分比
            # print(dqn_loss)
            
            print(f"Iteration {it}, Avg Locomotion Percentage: {loco_percentage * 100:.2f}%")
            print(f"Iteration {it}, Avg recovery Percentage: {reco_percentage * 100:.2f}%")
            print(f"Iteration {it}, Replay Buffer Size: {len(self.replay_buffer)}")
            print('Current mean reward', cur_reward_sum.mean())

            print('Current mean episode length', cur_episode_length.mean())
            print(f"Q-Learning Loss: {mean_dqn_loss}")


            locs = locals()
            wandb_dict = {
                "Iteration": it,
                "DQN_Loss": mean_dqn_loss,
                "Locomotion_Percentage": loco_percentage.item(),
                "Recovery_Percentage": reco_percentage.item(),
                "Replay_Buffer_Size": len(self.replay_buffer),
                "Mean_Reward": torch.mean(torch.tensor(rewbuffer, dtype=torch.float32, device=self.device)).item(),
                "Mean_Episode_Length": torch.mean(torch.tensor(lenbuffer, dtype=torch.float32, device=self.device)).item(),
                "Switch_penalty": torch.mean(torch.tensor(switch_penalty_buf, dtype=torch.float32, device=self.device)).item(),
                'Epsilon': self.epsilon
            }

            if locs.get('ep_infos') and len(locs['ep_infos']) > 0:  # 确保 ep_infos 存在且非空
                for key in locs['ep_infos'][0]:
                    infotensor = torch.empty(0, device=self.device)  # 初始化空张量
                    for ep_info in locs['ep_infos']:
                        # 将 ep_info[key] 转换为张量，处理标量和零维张量情况
                        value_tensor = torch.as_tensor(ep_info[key], device=self.device, dtype=torch.float32).flatten()
                        infotensor = torch.cat((infotensor, value_tensor))  # 拼接张量
                    value = infotensor.mean()  # 计算均值
                    # 根据 key 名称分类并记录到 wandb_dict
                    if "tracking" in key:
                        wandb_dict[f"Episode_rew_tracking/{key}"] = value
                    elif "curriculum" in key:
                        wandb_dict[f"Episode_curriculum/{key}"] = value
                    elif "terrain_level" in key:
                        wandb_dict[f"Episode_terrain_level/{key}"] = value
                    else:
                        wandb_dict[f"Episode_rew_regularization/{key}"] = value
    
            wandb.log(wandb_dict)


            # 每个学习迭代结束后保存 selector 模型
            if it < 2000:
                if (it + 1) % 500 == 0:  # 每10次迭代保存一次模型
                    selector_filename = f"selector_model_{it + 1}.pt"
                    self.save_selector(selector_filename)
                    # self.load_selector(selector_filename)  # 加载最新的模型
                    print(f"Selector model saved to {selector_filename}")
            else:
                if (it + 1) % 200 == 0:
                    selector_filename = f"selector_model_{it + 1}.pt"
                    self.save_selector(selector_filename)
                    print(f"Selector model saved to {selector_filename}")      
            ep_infos.clear()
