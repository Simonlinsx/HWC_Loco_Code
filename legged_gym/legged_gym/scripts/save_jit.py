import os, sys
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_mimic import Actor, get_activation
from rsl_rl.modules.estimator import Estimator
import argparse
import code
import shutil

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareRefNN(nn.Module):
    def __init__(self,  num_prop,
                        num_demo,
                        text_feat_input_dim,
                        text_feat_output_dim,
                        feat_hist_len,
                        n_decoder_out, 
                        num_labels,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        activation='elu',
                        encoder_hidden_dims=[256, 128, 64],
                        decoder_hidden_dims=[256, 128, 64],
                        num_latent=16,
                        ):
        super().__init__()

        self.text_feat_input_dim = text_feat_input_dim
        self.text_feat_output_dim = text_feat_output_dim
        self.feat_hist_len = feat_hist_len

        self.num_demo = num_demo

        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.n_decoder_out = n_decoder_out
        self.num_labels = num_labels
        num_obs = num_prop + num_hist * num_prop + num_labels
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, 
                           num_demo,
                           text_feat_input_dim,
                           text_feat_output_dim,
                           feat_hist_len,
                           num_actions, 
                           actor_hidden_dims, 
                           n_decoder_out, 
                           num_labels, 
                           num_hist, 
                           activation, tanh_encoder_output=tanh)

        self.estimator = Estimator(
            num_obs=num_prop,
            num_history=feat_hist_len,
            num_latent=num_latent,
            num_labels=num_labels,
            activation='elu',
            decoder_hidden_dims=decoder_hidden_dims,
            encoder_hidden_dims=encoder_hidden_dims,
        ).to('cpu')
        estimator_input = torch.ones(1, (feat_hist_len + 1) * num_prop)
        self.estimator = torch.jit.trace(self.estimator, estimator_input, strict=False)
        

    def forward(self, obs):
        # Keep the scripted estimator attached for downstream HRL code that
        # explicitly calls `policy.estimator(...)` after loading the JIT module.
        _ = self.estimator(obs[:, : self.text_feat_input_dim + self.num_prop])
        backbone_input = obs

        return self.actor(backbone_input, hist_encoding=True, eval=False)

def play(args):    
    load_run = "../../logs/h1/" + args.exptid
    checkpoint = args.checkpoint

    # Keep these dimensions aligned with the current H1 command policy config.
    n_labels = 3 + 1 + 1 + 4 + 1 + 19 * 2 + 19 * 2 + 5
    num_actions = 19
    
    # Current H1 command checkpoints were trained with gait features enabled.
    n_proprio = 3 + 2 + 2 + 19*3 + 3 - 2 + 3 + 2
    history_len = 11
    
    num_demo = 0
    feat_hist_len = 5
    text_feat_input_dim = feat_hist_len * n_proprio
    text_feat_output_dim = 16


    n_latent = 16
    n_decoder_out = n_labels + n_latent

    n_feature = text_feat_input_dim


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy = HardwareRefNN(n_proprio, 
                           num_demo,
                           text_feat_input_dim,
                           text_feat_output_dim,
                           feat_hist_len,
                           n_decoder_out,
                           n_labels,
                           history_len, 
                           num_actions, args.tanh).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'], strict=True)
    
    policy = policy.to(device)
    
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 1
        obs_input = torch.ones(num_envs, n_feature + n_proprio + n_decoder_out, device=device)
        print("obs_input shape: ", obs_input.shape)

        # Trace and save the Policy
        traced_policy = torch.jit.trace(policy, obs_input, strict=False)
        # traced_policy = torch.jit.script(policy)
        actor_save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-actor_jit.pt")
        traced_policy.save(actor_save_path)
        print("Saved traced_actor at ", os.path.abspath(actor_save_path))


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
