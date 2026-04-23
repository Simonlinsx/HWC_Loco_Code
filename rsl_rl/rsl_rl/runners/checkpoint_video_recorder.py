import os
import re
import subprocess
import sys
import warnings
from pathlib import Path


class CheckpointVideoRecorder:
    def __init__(self, runner):
        self.runner = runner
        self.cfg = runner.cfg
        self.enabled = bool(self.cfg.get("record_checkpoint_video", True))
        self.video_dir = os.path.join(runner.log_dir, "videos") if runner.log_dir is not None else None
        self.num_steps = int(self.cfg.get("checkpoint_video_num_steps", 300))
        self.clean_recovery_eval = bool(self.cfg.get("checkpoint_video_clean_recovery_eval", False))
        self.nodelay = bool(self.cfg.get("checkpoint_video_nodelay", False))
        self.play_script = Path(__file__).resolve().parents[3] / "legged_gym" / "legged_gym" / "scripts" / "play.py"
        self.active_process = None
        self.active_log_path = None
        self.active_video_dir = None

    def record(self, checkpoint_path):
        if not self.enabled or self.video_dir is None:
            return None
        if not self.play_script.exists():
            warnings.warn(f"Checkpoint video recorder could not find play.py at {self.play_script}")
            return None

        self._report_finished_process()
        if self.active_process is not None and self.active_process.poll() is None:
            warnings.warn(
                f"Skipping checkpoint video for {checkpoint_path} because a previous recording is still running. "
                f"See {self.active_log_path}."
            )
            return None

        checkpoint = self._parse_checkpoint_id(checkpoint_path)
        command = self._build_command(checkpoint_path, checkpoint)
        if command is None:
            return None

        os.makedirs(self.video_dir, exist_ok=True)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        self.active_log_path = os.path.join(self.video_dir, checkpoint_name + "_rollout.log")
        self.active_video_dir = self.video_dir

        env = os.environ.copy()
        with open(self.active_log_path, "w", encoding="utf-8") as log_file:
            self.active_process = subprocess.Popen(
                command,
                cwd=str(self.play_script.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(
            f"Started checkpoint rollout recording for {checkpoint_name} "
            f"(pid={self.active_process.pid}). Logs: {self.active_log_path}"
        )
        return self.active_video_dir

    def _report_finished_process(self):
        if self.active_process is None:
            return

        return_code = self.active_process.poll()
        if return_code is None:
            return

        if return_code == 0:
            print(f"Checkpoint rollout video saved under {self.active_video_dir}")
        else:
            warnings.warn(
                f"Checkpoint rollout recording failed with exit code {return_code}. "
                f"See {self.active_log_path}."
            )

        self.active_process = None
        self.active_log_path = None
        self.active_video_dir = None

    def _parse_checkpoint_id(self, checkpoint_path):
        match = re.search(r"model_(\d+)\.pt$", os.path.basename(checkpoint_path))
        if match is None:
            return -1
        return int(match.group(1))

    def _build_command(self, checkpoint_path, checkpoint):
        args = getattr(self.runner, "args", None)
        if args is None:
            warnings.warn("Checkpoint video recorder could not access training args; skipping video recording.")
            return None

        task = getattr(args, "task", None)
        if task is None:
            warnings.warn("Checkpoint video recorder could not infer task name; skipping video recording.")
            return None

        exptid = os.path.basename(self.runner.log_dir.rstrip(os.sep))
        proj_name = getattr(args, "proj_name", None) or os.path.basename(os.path.dirname(self.runner.log_dir.rstrip(os.sep)))
        motion_task = getattr(args, "motion_task", None) or getattr(self.runner.env.cfg.task, "motion_task", None)
        motion_name = getattr(args, "motion_name", None) or getattr(self.runner.env.cfg.motion, "motion_name", None)
        motion_type = getattr(args, "motion_type", None) or getattr(self.runner.env.cfg.motion, "motion_type", None)
        sim_device = getattr(args, "sim_device", None) or getattr(self.runner.env, "sim_device", "cuda:0")
        rl_device = getattr(args, "rl_device", None) or str(self.runner.device)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

        command = [
            sys.executable,
            str(self.play_script),
            exptid,
            "--task",
            task,
            "--proj_name",
            proj_name,
            "--sim_device",
            str(sim_device),
            "--rl_device",
            str(rl_device),
            "--checkpoint",
            str(checkpoint),
            "--record_video",
            "--headless",
            "--num_envs",
            "1",
            "--video_dir",
            self.video_dir,
            "--video_name_prefix",
            checkpoint_name,
            "--video_steps",
            str(self.num_steps),
        ]

        if motion_task is not None:
            command.extend(["--motion_task", str(motion_task)])
        if motion_name is not None:
            command.extend(["--motion_name", str(motion_name)])
        if motion_type is not None:
            command.extend(["--motion_type", str(motion_type)])
        if self.nodelay:
            command.append("--nodelay")
        if motion_task == "recovery" and self.clean_recovery_eval:
            command.append("--clean_recovery_eval")

        return command
