import torch
import torch.nn as nn
import torch.autograd
import pandas as pd
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
import os
from typing import Tuple
import time
import argparse

from uuv_baselines.uuv_gym import UUVEnvFMDP, EarlyStoppingCallback
from incremental_repair_utils import UUV_Control_NN, load_model_dict, dump_model_dict


class Control_NN_Aug(nn.Module):

    def __init__(self, layer_1_size=32, layer_2_size=32):
        super(Control_NN_Aug, self).__init__()
        self.fc1 = nn.Linear(3, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="training algo", default='ppo')
parser.add_argument("--steps", help="max total training steps, default 100000", default=100000)
parser.add_argument("--network", help="network yml file to be repaired", default=os.path.join('controllers', 'uuv_tanh_2_15_2x32_broken.yml'))
parser.add_argument("--sampled_result_path", help="path to sampling result csv", default='uuv_sampling_result.csv')
args = parser.parse_args()

if args.algo not in ['ppo', 'sac', 'a2c']:
    raise NotImplementedError


working_dir = 'uuv_baselines_fmdp_' + args.algo
if not os.path.exists(working_dir):
    os.mkdir(working_dir)
log_dir = os.path.join(working_dir, 'training_log.csv')


# Get all failed initial states
print('Obtaining all bad initial states to be repaired ...')
sampled_result_path = args.sampled_result_path
df_sample = pd.read_csv(sampled_result_path)
df_bad = df_sample[df_sample['result'] < 0.0]
bad_states = [(row['y'], row['h']) for index, row in df_bad.iterrows()]
env = UUVEnvFMDP(initial_states=bad_states)
env = Monitor(env, log_dir)

# Load pretrained weights into policy
net_path = args.network
net = load_model_dict(net_path)

# Migrate the net to augmented architecture
net_dict = net.state_dict()
net_aug = Control_NN_Aug()
net_aug_dict = net_aug.state_dict()

new_layer1_weights = torch.cat(
    (net_dict['fc1.weight'], torch.randn(net_aug.fc1.out_features, 1)), dim=1
)
net_aug_dict['fc1.weight'] = new_layer1_weights
net_aug_dict['fc1.bias'] = net_dict['fc1.bias']
net_aug_dict['fc2.weight'] = net_dict['fc2.weight']
net_aug_dict['fc2.bias'] = net_dict['fc2.bias']
net_aug_dict['fc3.weight'] = net_dict['fc3.weight']
net_aug_dict['fc3.bias'] = net_dict['fc3.bias']
net_aug.load_state_dict(net_aug_dict)


# Actor-critic wrapper over the policy network
class Control_NN_Aug_Wrapper(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 3,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = net_aug  # Load pretrained weights here

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


net_aug_wrapper = Control_NN_Aug_Wrapper()


# Custom policy classes to load pretrained weights
class UUVCustomActorCriticPolicyAug(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False  # Disable orthogonal initialization

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = net_aug_wrapper  # Load pretrained weights


class UUVCustomSACPolicyAug(SACPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.ortho_init = False  # Disable orthogonal initialization
        self.latent_pi = net_aug  # Load pretrained weights


print('Initializing policy model ...')
if args.algo == 'ppo':
    model = PPO(UUVCustomActorCriticPolicyAug, env, verbose=1)
elif args.algo == 'a2c':
    model = A2C(UUVCustomActorCriticPolicyAug, env, verbose=1)
elif args.algo == 'sac':
    model = SAC(UUVCustomSACPolicyAug, env, verbose=1)
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
    net_aug_repaired = model.policy.mlp_extractor.policy_net
elif args.algo == 'sac':
    net_aug_repaired = model.policy.latent_pi
else:
    raise NotImplementedError
save_path_aug = os.path.join(working_dir, 'repaired_net_aug_new.pth')
torch.save(net_aug_repaired, save_path_aug)
save_path_yml_aug = os.path.join(working_dir, 'repaired_net_aug_new.yml')
dump_model_dict(save_path_yml_aug, net_aug_repaired)
print('FMDP model saved')

# Extract the original architecture
net_repaired = UUV_Control_NN()
net_repaired_dict = net_repaired.state_dict()
net_aug_repaired_dict = net_aug_repaired.state_dict()

net_repaired_dict['fc1.weight'] = net_aug_repaired_dict['fc1.weight'][:, :-1]  # drop the additional params
net_repaired_dict['fc1.bias'] = net_aug_repaired_dict['fc1.bias']
net_repaired_dict['fc2.weight'] = net_aug_repaired_dict['fc2.weight']
net_repaired_dict['fc2.bias'] = net_aug_repaired_dict['fc2.bias']
net_repaired_dict['fc3.weight'] = net_aug_repaired_dict['fc3.weight']
net_repaired_dict['fc3.bias'] = net_aug_repaired_dict['fc3.bias']
net_repaired.load_state_dict(net_repaired_dict)

save_path = os.path.join(working_dir, 'repaired_net_new.pth')
torch.save(net_repaired, save_path)
save_path_yml = os.path.join(working_dir, 'repaired_net_new.yml')
dump_model_dict(save_path_yml, net_repaired)


