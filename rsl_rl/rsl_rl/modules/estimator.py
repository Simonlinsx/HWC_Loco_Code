from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F

# class Estimator(nn.Module):
#     def __init__(self,  input_dim,
#                         output_dim,
#                         hidden_dims=[256, 128, 64],
#                         activation="elu",
#                         **kwargs):
#         super(Estimator, self).__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         activation = get_activation(activation)
#         estimator_layers = []
#         estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
#         estimator_layers.append(activation)
#         for l in range(len(hidden_dims)):
#             if l == len(hidden_dims) - 1:
#                 estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
#             else:
#                 estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
#                 estimator_layers.append(activation)
#         # estimator_layers.append(nn.Tanh())
#         self.estimator = nn.Sequential(*estimator_layers)
    
#     def forward(self, input):
#         return self.estimator(input)
    
#     def inference(self, input):
#         with torch.no_grad():
#             return self.estimator(input)



# VAE + improved KL
class Estimator(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 num_labels,
                 activation='elu',
                 decoder_hidden_dims=[512, 256, 128], 
                 encoder_hidden_dims=[512, 256]):
        super(Estimator, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent
        self.num_labels = num_labels

        # Posterior encoder: history + current observation -> latent and labels
        self.encoder = MLPHistoryEncoder(
            num_obs=num_obs,
            num_history=num_history + 1,
            num_latent=num_latent * 4,
            activation=activation,
            adaptation_module_branch_hidden_dims=encoder_hidden_dims,
        )
        self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        self.latent_var = nn.Linear(num_latent * 4, num_latent)
        self.label_mu = nn.Linear(num_latent * 4, num_labels)

        # History-only prior for a meaningful KL term
        self.prior_mu = nn.Linear(num_obs * num_history, num_latent)
        self.prior_var = nn.Linear(num_obs * num_history, num_latent)

        # Decoder predicts the future proprioceptive observation
        modules = []
        activation_fn = get_activation(activation)
        decoder_input_dim = num_latent + num_labels
        modules.extend(
            [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
            activation_fn]
        )
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l], num_obs))
            else:
                modules.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                modules.append(activation_fn)
        self.decoder = nn.Sequential(*modules)

    def _reshape_history(self, obs_history):
        if obs_history.dim() == 2:
            return obs_history.reshape(-1, self.num_history, self.num_obs)
        return obs_history

    def _reshape_current(self, current_obs):
        if current_obs.dim() == 1:
            return current_obs.unsqueeze(0)
        return current_obs

    def _prepare_inputs(self, obs_history, current_obs=None):
        if current_obs is None and obs_history.dim() == 2 and obs_history.shape[1] == self.num_obs * (self.num_history + 1):
            full_context = obs_history.reshape(-1, self.num_history + 1, self.num_obs)
            return full_context[:, : self.num_history, :], full_context[:, -1, :]

        obs_history = self._reshape_history(obs_history)
        if current_obs is None:
            current_obs = obs_history[:, -1, :]
        else:
            current_obs = self._reshape_current(current_obs)
        return obs_history, current_obs

    def encode(self, obs_history, current_obs=None):
        obs_history, current_obs = self._prepare_inputs(obs_history, current_obs)
        posterior_context = torch.cat((obs_history, current_obs.unsqueeze(1)), dim=1)
        encoded = self.encoder(posterior_context)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        label_mu = self.label_mu(encoded)
        return [latent_mu, latent_var, label_mu]

    def compute_prior(self, obs_history):
        obs_history, _ = self._prepare_inputs(obs_history, None)
        prior_input = obs_history.reshape(obs_history.shape[0], -1)
        prior_mu = self.prior_mu(prior_input)
        prior_var = self.prior_var(prior_input)

        return prior_mu, prior_var

    def decode(self, z, labels):
        input = torch.cat([z, labels], dim=1)
        output = self.decoder(input)
        return output

    def forward(self, obs_history, current_obs=None):
        latent_mu, latent_var, label_mu = self.encode(obs_history, current_obs)
        z = self.reparameterize(latent_mu, latent_var)

        # Compute prior distribution
        prior_mu, prior_var = self.compute_prior(obs_history)

        return [z, label_mu], [latent_mu, latent_var, label_mu, prior_mu, prior_var]

    # def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     if torch.cuda.is_available():
    #         eps = torch.randn_like(std).cuda()
    #     else:
    #         eps = torch.randn_like(std)
    #     return eps * std + mu


    def loss_fn(self, obs_history, current_obs, future_obs, future_labels, dones=None, kld_weight=1.0):
        estimation, latent_params = self.forward(obs_history, current_obs)
        z, labels = estimation
        latent_mu, latent_var, label_mu, prior_mu, prior_var = latent_params

        # Reconstruction loss
        recons = self.decode(z, labels)
        recons_loss = F.mse_loss(recons, future_obs, reduction='none').mean(-1)

        # Predict the deterministic label mean for state/domain labels
        label_loss = F.mse_loss(label_mu, future_labels, reduction='none').mean(-1)

        # KL divergence loss
        kld_loss = 0.5 * torch.sum(
            prior_var - latent_var +
            (latent_var.exp() + (latent_mu - prior_mu).pow(2)) / prior_var.exp() - 1,
            dim=1
        )

        # Total loss
        loss = recons_loss + label_loss + kld_weight * kld_loss
        if dones is not None:
            valid_mask = (~dones.view(-1).bool()).float()
            valid_scale = valid_mask.numel() / valid_mask.sum().clamp(min=1.0)
            loss = loss * valid_mask * valid_scale
            recons_loss = recons_loss * valid_mask * valid_scale
            label_loss = label_loss * valid_mask * valid_scale
            kld_loss = kld_loss * valid_mask * valid_scale
        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'label_loss': label_loss,
            'kld_loss': kld_loss,
        }

    def sample(self, obs_history, current_obs=None):
        estimation, _ = self.forward(obs_history, current_obs)
        return estimation

    def inference(self, obs_history, current_obs=None):
        _, latent_params = self.forward(obs_history, current_obs)
        latent_mu, latent_var, label_mu, _, _ = latent_params
        return [latent_mu, label_mu]



# # For DreamWaQ
# # VAE structure
# class Estimator(nn.Module):
#     def __init__(self, 
#                  num_obs,
#                  num_history,
#                  num_latent,
#                  activation = 'elu',
#                  decoder_hidden_dims = [512, 256, 128],encoder_hidden_dims = [256, 64]):
#         super(Estimator, self).__init__()
#         self.num_obs = num_obs
#         self.num_history = num_history
#         self.num_latent = num_latent


#         # Build Encoder
    
#         self.encoder = MLPHistoryEncoder(
#             num_obs = num_obs,
#             num_history=num_history,
#             num_latent=num_latent * 4,
#             activation=activation,
#             adaptation_module_branch_hidden_dims=encoder_hidden_dims,
#         )
#         self.latent_mu = nn.Linear(num_latent * 4, num_latent)
#         self.latent_var = nn.Linear(num_latent * 4, num_latent)
        
#         self.vel_mu = nn.Linear(num_latent * 4, 3)
#         self.vel_var = nn.Linear(num_latent * 4, 3)

#         # Build Decoder
#         modules = []
#         activation_fn = get_activation(activation)
#         decoder_input_dim = num_latent + 3
#         modules.extend(
#             [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
#             activation_fn]
#             )
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 modules.append(nn.Linear(decoder_hidden_dims[l],num_obs))
#             else:
#                 modules.append(nn.Linear(decoder_hidden_dims[l],decoder_hidden_dims[l + 1]))
#                 modules.append(activation_fn)
#         self.decoder = nn.Sequential(*modules)

#     def encode(self,obs_history):
#         encoded = self.encoder(obs_history)
#         latent_mu = self.latent_mu(encoded)
#         latent_var = self.latent_var(encoded)
#         vel_mu = self.vel_mu(encoded)
#         vel_var = self.vel_var(encoded)
#         return [latent_mu, latent_var, vel_mu, vel_var]

#     def decode(self,z,v):
#         input = torch.cat([z,v], dim = 1)
#         output = self.decoder(input)
#         return output

#     def forward(self,obs_history):
#         latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
#         z = self.reparameterize(latent_mu, latent_var)
#         vel = self.reparameterize(vel_mu, vel_var)
#         return [z,vel],[latent_mu, latent_var, vel_mu, vel_var]
    
#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """
#         Will a single z be enough ti compute the expectation
#         for the loss??
#         :param mu: (Tensor) Mean of the latent Gaussian
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian
#         :return:
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
    
#     def loss_fn(self,obs_history, next_obs, vel,kld_weight = 1.0):
#         estimation, latent_params = self.forward(obs_history)
#         z, v = estimation
#         latent_mu, latent_var, vel_mu, vel_var = latent_params 
#         # Reconstruction loss
#         recons = self.decode(z,vel)
#         recons_loss =F.mse_loss(recons, next_obs,reduction='none').mean(-1)
#         # Supervised loss
#         vel_loss = F.mse_loss(v, vel,reduction='none').mean(-1)

#         kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim = 1)

#         loss = recons_loss + vel_loss + kld_weight * kld_loss
#         return {
#             'loss': loss,
#             'recons_loss': recons_loss,
#             'vel_loss': vel_loss,
#             'kld_loss': kld_loss,
#         }
#     def sample(self,obs_history):
#         estimation, _ = self.forward(obs_history)
#         return estimation
    
#     def inference(self,obs_history):
#         _, latent_params = self.forward(obs_history)
#         latent_mu, latent_var, vel_mu, vel_var = latent_params
#         return [latent_mu, vel_mu]




class MLPHistoryEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 adaptation_module_branch_hidden_dims = [256, 128],):
        super(MLPHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        input_size = num_obs * num_history
        output_size = num_latent

        activation = get_activation(activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], output_size))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.encoder = nn.Sequential(*adaptation_module_layers)
    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T , obs_dim)
        """
        bs = obs_history.shape[0]
        T = self.num_history
        output = self.encoder(obs_history.reshape(bs, -1))
        return output



# GRU Encoder
# class Estimator(nn.Module):
#     def __init__(self, 
#                  num_obs,
#                  num_history,
#                  num_latent,
#                  activation='elu',
#                  decoder_hidden_dims=[512, 256, 128], 
#                  gru_hidden_size=256,  # 新增：GRU 的隐藏层大小
#                  gru_num_layers=1,  # 新增：GRU 的层数
#                  ):
#         super(Estimator, self).__init__()
#         self.num_obs = num_obs
#         self.num_history = num_history
#         self.num_latent = num_latent
#         # Build Encoder: 使用 GRU 替换原有的 MLPHistoryEncoder
#         self.encoder = nn.GRU(
#             input_size=num_obs,
#             hidden_size=gru_hidden_size,
#             num_layers=gru_num_layers,
#             batch_first=True
#         )
#         self.latent_mu = nn.Linear(gru_hidden_size, num_latent)
#         self.latent_var = nn.Linear(gru_hidden_size, num_latent)

#         self.vel_mu = nn.Linear(gru_hidden_size, 4)
#         self.vel_var = nn.Linear(gru_hidden_size, 4)

#         # Build Prior
#         self.prior_mu = nn.Linear(num_obs, num_latent)
#         self.prior_var = nn.Linear(num_obs, num_latent)

#         # Build Decoder
#         modules = []
#         activation_fn = get_activation(activation)
#         decoder_input_dim = num_latent + 4
#         modules.extend(
#             [nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
#              activation_fn]
#         )
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 modules.append(nn.Linear(decoder_hidden_dims[l], num_obs))
#             else:
#                 modules.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
#                 modules.append(activation_fn)
#         self.decoder = nn.Sequential(*modules)

#     def encode(self, obs_history):
#         # 使用 GRU 进行编码
#         _, hidden = self.encoder(obs_history)
#         # 取最后一层的隐藏状态作为编码结果
#         encoded = hidden[-1]
#         latent_mu = self.latent_mu(encoded)
#         latent_var = self.latent_var(encoded)
#         vel_mu = self.vel_mu(encoded)
#         vel_var = self.vel_var(encoded)
#         return [latent_mu, latent_var, vel_mu, vel_var]

#     def compute_prior(self, obs_history):
#         # Use the most recent observation for prior distribution
#         last_obs = obs_history[:, -self.num_obs:]  # Assuming obs_history has shape (batch_size, num_history, num_obs)
#         prior_mu = self.prior_mu(last_obs)
#         prior_var = self.prior_var(last_obs)
#         return prior_mu, prior_var

#     def decode(self, z, v):
#         input = torch.cat([z, v], dim=1)
#         output = self.decoder(input)
#         return output

#     def forward(self, obs_history):
#         latent_mu, latent_var, vel_mu, vel_var = self.encode(obs_history)
#         z = self.reparameterize(latent_mu, latent_var)
#         vel = self.reparameterize(vel_mu, vel_var)

#         # Compute prior distribution
#         prior_mu, prior_var = self.compute_prior(obs_history)

#         return [z, vel], [latent_mu, latent_var, vel_mu, vel_var, prior_mu, prior_var]

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu


#     def loss_fn(self, obs_history, next_obs, vel, kld_weight=1.0):
#         estimation, latent_params = self.forward(obs_history)
#         z, v = estimation
#         latent_mu, latent_var, vel_mu, vel_var, prior_mu, prior_var = latent_params

#         # Reconstruction loss
#         recons = self.decode(z, v)
#         recons_loss = F.mse_loss(recons, next_obs, reduction='none').mean(-1)

#         # Supervised velocity loss
#         vel_loss = F.mse_loss(v, vel, reduction='none').mean(-1)

#         # KL divergence loss
#         kld_loss = 0.5 * torch.sum(
#             prior_var - latent_var +
#             (latent_var.exp() + (latent_mu - prior_mu).pow(2)) / prior_var.exp() - 1,
#             dim=1
#         )

#         # Total loss
#         loss = recons_loss + vel_loss + kld_weight * kld_loss
#         return {
#             'loss': loss,
#             'recons_loss': recons_loss,
#             'vel_loss': vel_loss,
#             'kld_loss': kld_loss,
#         }

#     def sample(self, obs_history):
#         estimation, _ = self.forward(obs_history)
#         return estimation


#     def inference(self, obs_history):
#         _, latent_params = self.forward(obs_history)
#         latent_mu, latent_var, vel_mu, vel_var, _, _ = latent_params
#         return [latent_mu, vel_mu]




# Transformer

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Estimator(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], num_heads=2, num_layers=2, dropout=0.1):
#         super(Estimator, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Embedding layer to project input to the transformer dimension
#         self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        
#         # Transformer encoder layer
#         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dims[0], nhead=num_heads, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Output layer
#         self.output_layer = nn.Linear(hidden_dims[1], output_dim)
    
#     def forward(self, input):
#         # Assuming input shape is (batch_size, seq_len, input_dim)
        
#         # Project input to transformer dimension
#         x = self.input_proj(input)  # Shape: (batch_size, seq_len, hidden_dim)
        
        
#         # Permute to match Transformer input shape requirements (seq_len, batch_size, hidden_dim)
#         x = x.permute(1, 0, 2)
        
#         # Pass through Transformer encoder
#         x = self.transformer_encoder(x)  # Shape: (seq_len, batch_size, hidden_dim)
        
#         # Take the mean across the sequence dimension (global average pooling)
#         x = x.mean(dim=0)  # Shape: (batch_size, hidden_dim)
        
#         # Output layer
#         output = self.output_layer(x)  # Shape: (batch_size, output_dim)
#         return output
    
#     def inference(self, input):
#         with torch.no_grad():
#             return self.forward(input)


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Compute positional encodings once in log space
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
        
#         # Register as buffer to avoid updating this in backpropagation
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         # Add positional encoding to input x
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


class Discriminator(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        # init_weight(self.q, initializer="xavier uniform")
        # self.q.bias.data.zero_()

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)

class DiscriminatorLSD(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorLSD, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(spectral_norm(nn.Linear(n_states, hidden_dims[0])))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], n_skills)))
            else:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)
        
class DiscriminatorContDIAYN(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorContDIAYN, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)

    def forward(self, states):
        return torch.nn.functional.normalize(self.discriminator(states))

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)
