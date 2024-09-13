import torch.nn as nn
import torch.autograd
import pandas as pd
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy
import os
from typing import Tuple
import time
import argparse

from mc_baselines.mc_gym import MCEnv, EarlyStoppingCallback
from incremental_repair_utils import MC_Control_NN, load_model_dict, dump_model_dict


parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="training algo", default='ppo')
parser.add_argument("--steps", help="training total time steps, default 100000", default=100)
parser.add_argument("--network", help="network yml file to be repaired", default=os.path.join('controllers', 'mc_sig_2x16_broken.yml'))
parser.add_argument("--sampled_result_path", help="path to sampling result csv", default='mc_sampling_result.csv')
args = parser.parse_args()

if args.algo not in ['ppo', 'sac', 'a2c']:
    raise NotImplementedError


working_dir = 'mc_baselines_stlgym_' + args.algo
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
log_dir = os.path.join(working_dir, 'training_log.csv')


# Get all failed initial states
print('Obtaining all bad initial states to be repaired ...')
sampled_result_path = args.sampled_result_path
df_sample = pd.read_csv(sampled_result_path)
df_bad = df_sample[df_sample['result'] < 0.0]
bad_states = [(row['pos'], row['vel']) for index, row in df_bad.iterrows()]
env = MCEnv(initial_states=bad_states)
env = Monitor(env, log_dir)

# Load pretrained weights into policy
net_path = args.network
net = load_model_dict(net_path)


# Actor-critic wrapper over the policy network
class Control_NN_Wrapper(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 2,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = net  # Load pretrained weights here

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


net_wrapper = Control_NN_Wrapper()


# Custom policy classes to load pretrained weights
class MCCustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False  # Disable orthogonal initialization

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = net_wrapper  # Load pretrained weights


class MCCustomSACPolicy(SACPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False  # Disable orthogonal initialization
        self.latent_pi = net  # Load pretrained weights


class MCCustomTD3Policy(TD3Policy):

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False  # Disable orthogonal initialization


print('Initializing policy model ...')
if args.algo == 'ppo':
    model = PPO(MCCustomActorCriticPolicy, env, verbose=1)
elif args.algo == 'sac':
    model = SAC(MCCustomSACPolicy, env, verbose=1)
elif args.algo == 'a2c':
    model = A2C(MCCustomActorCriticPolicy, env, verbose=1)
else:
    raise NotImplementedError

print('Model policy architecture:')
if args.algo in ['ppo', 'a2c']:
    print(model.policy.mlp_extractor.policy_net)
elif args.algo == 'sac':
    print(model.policy.latent_pi)
else:
    raise NotImplementedError

callback = EarlyStoppingCallback()

# Retrain on all failed initial states
print('Begin retraining on failed initial states ...')
start_time = time.time()
model.learn(total_timesteps=int(args.steps), callback=callback)
print(f'Training time = {time.time() - start_time}')


# Save the whole thing first
model.save(os.path.join(working_dir, 'agent_new.zip'))
print('Whole RL agent saved')

# Save retrained model
if args.algo in ['ppo', 'a2c']:
    repaired_net = model.policy.mlp_extractor.policy_net
elif args.algo == 'sac':
    repaired_net = model.policy.latent_pi
else:
    raise NotImplementedError
save_path = os.path.join(working_dir, 'repaired_net_new.pth')
torch.save(repaired_net, save_path)
save_path_yml = os.path.join(working_dir, 'repaired_net_new.yml')
dump_model_dict(save_path_yml, repaired_net)

print('STLGym model saved')