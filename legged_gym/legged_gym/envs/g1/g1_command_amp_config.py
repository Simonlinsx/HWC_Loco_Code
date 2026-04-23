from legged_gym.envs.g1.g1_command_config import G1CommandCfg, G1CommandCfgPPO

class G1CommandAMPCfg( G1CommandCfg ):
    class amp():
        num_obs_steps = 2
        num_obs_per_step = 23 + 3 + 3 + 3 + 12*3 

class G1CommandAMPCfgPPO( G1CommandCfgPPO ):
    class runner( G1CommandCfgPPO.runner ):
        runner_class_name = "OnPolicyRunnerMimicAMP"
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
    
    class amp():
        amp_input_dim = G1CommandAMPCfg.amp.num_obs_steps * G1CommandAMPCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [1024, 512]

        amp_replay_buffer_size = 1000000
        amp_demo_buffer_size = 200000
        amp_demo_fetch_batch_size = 1000
        amp_learn_batch_size = 1024
        amp_learning_rate = 1.e-4

        amp_reward_coef = 0.8    # Is it too high?  0.2   0.4  

        amp_grad_pen = 5