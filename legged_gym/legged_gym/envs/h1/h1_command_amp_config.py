from legged_gym.envs.h1.h1_command_config import H1CommandCfg, H1CommandCfgPPO

class H1CommandAMPCfg( H1CommandCfg ):
    class amp():
        num_obs_steps = 2
        num_obs_per_step = 19 + 3 + 3 + 3 + 12*3


class H1CommandAMPCfgPPO( H1CommandCfgPPO ):
    class runner( H1CommandCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimicAMP"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
    
    class amp():
        amp_input_dim = H1CommandAMPCfg.amp.num_obs_steps * H1CommandAMPCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [1024, 512]

        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 512
        amp_learn_batch_size = 4096
        amp_learning_rate = 1.e-4

        amp_reward_coef = 0.5    # Is it too high?  0.5  0.4  1.0。
        amp_grad_pen = 5