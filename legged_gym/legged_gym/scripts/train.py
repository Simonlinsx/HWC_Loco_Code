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

import numpy as np
import argparse
import os

_MOTION_TASK_ENV_VAR = "LEGGED_GYM_MOTION_TASK"


def _apply_motion_task_cli_override():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--motion_task", choices=("walk", "recovery"))
    early_args, _ = parser.parse_known_args()
    if early_args.motion_task is not None:
        os.environ[_MOTION_TASK_ENV_VAR] = early_args.motion_task


_apply_motion_task_cli_override()

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb
import random

def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(args):
    random_seed = args.seed if getattr(args, "seed", None) is not None else 42
    args.seed = random_seed
    set_random_seed(random_seed)
    print(f"Random seed set to: {random_seed}")

    if "g1" in args.task:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "g1", args.exptid)
    elif "h1" in args.task:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "h1", args.exptid)
    else:
        log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)

    os.makedirs(log_pth, exist_ok=True)
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 5
        args.num_envs = 64
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"

    wandb.init(
        project=args.proj_name,
        name=args.exptid,
        entity=args.entity if args.entity else None,
        mode=mode,
        dir=os.path.join(LEGGED_GYM_ROOT_DIR, "logs"),
    )

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # Log configs immediately
    args = get_args()

    if getattr(args, "seed", None) is None:
        args.seed = 42
        
    train(args)
