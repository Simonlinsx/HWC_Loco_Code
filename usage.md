
## Installation ##
```bash
conda create -n humanoid python=3.8
conda activate humanoid
cd
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:chengxuxin/expressive_humanoid.git
cd expressive_humanoid
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd ~/expressive_humanoid/rsl_rl && pip install -e .
cd ~/expressive_humanoid/legged_gym && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown
```
Next install fbx. Follow the instructions [here](https://github.com/nv-tlabs/ASE/issues/61).

## Prepare dataset
1. Download from [here](https://drive.google.com/file/d/1m3JRBox51cjV4CbiKIcVXR6eKQBjaAhO/view?usp=sharing) and extract the zip file to `ASE/ase/poselib/data/cmu_fbx_all` that contains all `.fbx` files.

2. Gnerate `.yaml` file for the motions you want to use. 
```bash
cd ASE/ase/poselib
python parse_cmu_mocap_all.py
```
This step is not mandatory because the `.yaml` file is already generated. But if you want to add more motions, you can use this script to generate the `.yaml` file.

3. Import motions 
```bash
cd ASE/ase/poselib
python fbx_importer_all.py
```
This will import all motions in CMU Mocap dataset into `ASE/ase/poselib/data/npy`.

4. Retarget motions
```bash
cd ASE/ase/poselib
mkdir pkl retarget_npy
python retarget_motion_h1_all.py
```
This will retarget all motions in `ASE/ase/poselib/data/npy` to `ASE/ase/poselib/data/retarget_npy`.

5. Gnerate keybody positions

This step will require running simulation to extract more precise key body positions. 
```bash
cd legged_gym/legged_gym/scripts
python train.py debug --task h1_view --motion_name motions_debug.yaml --debug
```
Train for 1 iteration and kill the program to have a dummy model to load. 
```bash
python play.py debug --task h1_view --motion_name motions_autogen_all.yaml
```
It is recommended to use `motions_autogen_all.yaml` at the first time, so that later if you have a subset it is not neccessary to regenerate keybody positions. This will generate keybody positions to `ASE/ase/poselib/data/retarget_npy`.
Set wandb asset: 

## Training

### Goal Tracking Policy


### Generate extreme initialization set



### Recovery Policy



### Selector Policy



## Evaluation








<!-- # No AMP
python train.py exp_name --task h1_command --motion_name motions_autogen_human_walk.yaml --motion_type yaml --sim_device cuda:0 --rl_device cuda:0

# With AMP
python train.py exp_name --task h1_command_amp --motion_name motions_autogen_human_walk.yaml --motion_type yaml --sim_device cuda:0 --rl_device cuda:0 --seed 2



### Training  
## H1
python train.py h1_amp --task h1_command_amp --motion_name motions_autogen_human_walk_and_run.yaml --motion_type yaml --sim_device cuda:0 --rl_device cuda:0 --seed 42 --headless

## G1
python train.py exp_name --task g1_command_amp --motion_name motions_autogen_human_walk_and_run_g1.yaml --motion_type yaml --sim_device cuda:0 --rl_device cuda:0 --seed 42 --headless

 -->


## Goal Tracking    need to motion_task = 'walk'    # walk
# Goal Tracking
CUDA_VISIBLE_DEVICES=2 python train.py goal_tracking_v2 \
  --task h1_command_amp \
  --motion_task walk \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless

CUDA_VISIBLE_DEVICES=2 python train.py goal_tracking_v3 \
  --task h1_command_amp \
  --motion_task walk \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py goal_tracking_v4 \
  --task h1_command_amp \
  --motion_task walk \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=2 python train.py goal_tracking_v5 \
  --task h1_command_amp \
  --motion_task walk \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=2 python train.py goal_tracking_v6 \
  --task h1_command_amp \
  --motion_task walk \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

## Recovery    need to motion_task = 'recovery'    # recovery
# Recovery
CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v2 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless


CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v3 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=0 python train.py recovery_track_cost_v4 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v6\
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v7\
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=0 python train.py recovery_track_cost_v8\
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=0 python train.py recovery_track_cost_v9\
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v11 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v12 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v13 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300



CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v15 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v16 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v17 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v18 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v19 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v22 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=0 python train.py recovery_track_cost_v21 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v22 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v23 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v24 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v26 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --motion_type yaml \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --seed 42 \
  --headless \
  --record_checkpoint_video \
  --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v27   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300

CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v28   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=2 python train.py recovery_track_cost_v29   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v30   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v31   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300



CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v32   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v33   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300



CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v34   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v35   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=1 python train.py recovery_track_cost_v42   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300


CUDA_VISIBLE_DEVICES=3 python train.py recovery_track_cost_v42_wo_zmp   --task h1_command   --motion_task recovery   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --headless   --record_checkpoint_video   --checkpoint_video_num_steps 300




## Selector
CUDA_VISIBLE_DEVICES=3 python train_selector.py selector_h1_v1   --task h1_selector   --motion_task walk   --motion_name motions_autogen_human_walk_and_run.yaml   --motion_type yaml   --sim_device cuda:0   --rl_device cuda:0   --seed 42   --proj_name h1   --loco_jit /data1/linsixu/HWC_Loco/legged_gym/logs/h1/goal_tracking/traced/goal_tracking-28000-actor_jit.pt   --reco_jit /data1/linsixu/HWC_Loco/legged_gym/logs/h1/recovery/traced/recovery-30800-actor_jit.pt



# goal tracking
cd /data1/linsixu/HWC_Loco/legged_gym/legged_gym/scripts
CUDA_VISIBLE_DEVICES=2 python play.py goal_tracking_v2 \
  --task h1_command_amp \
  --motion_task walk \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --proj_name h1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --checkpoint 20000 \
  --record_video \
  --headless



# recovery
cd /data1/linsixu/HWC_Loco/legged_gym/legged_gym/scripts
CUDA_VISIBLE_DEVICES=2 python play.py recovery_track_cost_v2 \
  --task h1_command \
  --motion_task recovery \
  --motion_name motions_autogen_human_walk_and_run.yaml \
  --proj_name h1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --checkpoint 30800 \
  --record_video \
  --headless




cd /data1/linsixu/HWC_Loco/legged_gym/legged_gym/scripts


CUDA_VISIBLE_DEVICES=2 python play_selector_jit.py selector_eval \
  --task h1_selector \
  --motion_task walk \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --headless \
  --record_video \
  --num_envs 1 \
  --selector_path /data1/linsixu/HWC_Loco/legged_gym/logs/h1/selector_h1_v1/selector_model_5800.pt \
  --loco_jit /data1/linsixu/HWC_Loco/legged_gym/logs/h1/goal_tracking/traced/goal_tracking-28000-actor_jit.pt \
  --reco_jit /data1/linsixu/HWC_Loco/legged_gym/logs/h1/recovery/traced/recovery-30800-actor_jit.pt