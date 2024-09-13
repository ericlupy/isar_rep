import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback


class MCEnv(gym.Env):

    def __init__(self, initial_states):

        self.initial_states = initial_states  # bad states to be repaired by retraining
        self.n_initial_states = len(initial_states)

        # dynamics parameters
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = 0
        self.power = 0.0015 # original power is 0.0015, reduced power is 0.0012, 0.0010 is complete trash and succeeds very rarely; ended up not changing it
        #self.steepness = 0 # SELECTED IN RESET # 0.0035 # originally 0.0025, challenging = 0.0035
        self.steepness_vals = [0.0025] # [0.0025, 0.0035]
        self.steepness_probs = [1] # [0.5, 0.5]
        self.POSNOISESTD = 0.0 # 0.001
        self.VELNOISESTD = 0.0 # 0.0001

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        # noise functions
        self.noise_pos = lambda pos, vel : self.truec*vel
        self.noise_vel = lambda pos, vel : self.trued*pos
        self.noise_pos_gen = lambda pos, vel, c: c*vel
        self.noise_vel_gen = lambda pos, vel, d: d*pos

        self.cur_episode = -1

        self.seed() # seed 1 is pretty bad, poor convergence
                    # seed 2 is worse
                    # 5-11 are kinda accurate but late
                    # seed 9 is accurate
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed_saved = seed  # self.np_random.get_state()[1][0]
        return [seed]

    def reset(self):

        self.cur_episode += 1

        # randomly pick the true simulation parameters
        # self.truec = self.np_random.uniform(low=self.TRUECLEFT, high=self.TRUECRIGHT)
        # self.trued = self.np_random.uniform(low=self.TRUEDLEFT, high=self.TRUEDRIGHT)
        self.truec = 0.0
        self.trued = 0.0

        # self.trueinitpos = self.np_random.uniform(low=self.INITPOSLEFT, high=self.INITPOSRIGHT)
        # self.trueinitvel = self.np_random.uniform(low=self.INITVELLEFT, high=self.INITVELRIGHT)
        self.trueinitpos = self.initial_states[self.cur_episode % self.n_initial_states][0]
        self.trueinitvel = self.initial_states[self.cur_episode % self.n_initial_states][1]

        # self.steepness = self.np_random.choice(self.steepness_vals, 1, p=self.steepness_probs)[0]  # originally 0.0025, challenging = 0.0035
        self.steepness = 0.0025

        # initialize the Gym-related vars
        self.state = np.array([self.trueinitpos, self.trueinitvel])
        self.time = 0

        # Remember the initial observation:
        self.init_pos_obs = self.state[0] + self.noise_pos(self.state[0], self.state[1])
        self.init_vel_obs = self.state[1] + self.noise_vel(self.state[0], self.state[1])

        # Initialize particles
        # self.weights = np.array([1. / self.PARTCT] * self.PARTCT)
        # self.particles_c = self.np_random.uniform(low=self.PFINITCLEFT, high=self.PFINITCRIGHT, size=self.PARTCT)
        # self.particles_d = self.np_random.uniform(low=self.PFINITDLEFT, high=self.PFINITDRIGHT, size=self.PARTCT)
        # self.particles_d = self.np_random.uniform(low=self.trued, high=self.trued, size=self.PARTCT)

        # Create initial GTs from initial observations
        # self.part_pos_gt, self.part_vel_gt = self.obs_to_true(self.init_pos_obs, self.init_vel_obs,self.particles_c, self.particles_d)

        # returning the observation
        return np.array([self.init_pos_obs, self.init_vel_obs])

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        # proc_noise = self.np_random.normal([0, 0], [self.POSNOISESTD, self.VELNOISESTD])
        proc_noise = np.array([0.0, 0.0])

        # GT state propagation
        position, velocity = self.model_step(position, velocity, force, proc_noise)
        self.state = np.array([position, velocity])

        # updating the simulation state
        done = bool((position >= self.goal_position and velocity >= self.goal_velocity) or self.time > 110)
        reward = 0
        if position >= 0.45:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        # new observed values
        obs_pos = position + self.noise_pos(position, velocity)
        obs_vel = velocity + self.noise_vel(position, velocity)
        obs_state = np.array([obs_pos, obs_vel])

        self.time += 1

        obs_state = obs_state.reshape((2,))

        return obs_state, reward, done, {}

    def model_step(self, oldpos, oldvel, act, proc_noise):
        # updating state values
        newvel = oldvel + act*self.power - self.steepness * math.cos(3*oldpos) + proc_noise[1]
        if (newvel > self.max_speed): newvel = np.array([self.max_speed])
        if (newvel < -self.max_speed): newvel = np.array([-self.max_speed])
        newpos = oldpos + newvel + proc_noise[0]
        if (newpos > self.max_position): newpos = np.array([self.max_position])
        if (newpos < self.min_position): newpos = np.array([self.min_position])
        if (newpos == self.min_position and newvel<0): newvel = np.array([0])
        return newpos, newvel


def get_flag_state(pos):

    if -0.7 <= pos <= 0.3:  # in the valley
        return 0
    elif pos < -0.7:  # accumulated momentum
        return 1
    else:  # on hill
        return 2


class MCEnvFMDP(gym.Env):

    def __init__(self, initial_states):

        self.initial_states = initial_states  # bad states to be repaired by retraining
        self.n_initial_states = len(initial_states)
        self.flag_state = 0

        # dynamics parameters
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = 0
        self.power = 0.0015 # original power is 0.0015, reduced power is 0.0012, 0.0010 is complete trash and succeeds very rarely; ended up not changing it
        #self.steepness = 0 # SELECTED IN RESET # 0.0035 # originally 0.0025, challenging = 0.0035
        self.steepness_vals = [0.0025] # [0.0025, 0.0035]
        self.steepness_probs = [1] # [0.5, 0.5]
        self.POSNOISESTD = 0.0 # 0.001
        self.VELNOISESTD = 0.0 # 0.0001

        self.low_state = np.array([self.min_position, -self.max_speed, 0])  # add flag state 0, 1, 2
        self.high_state = np.array([self.max_position, self.max_speed, 2])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        # noise functions
        self.noise_pos = lambda pos, vel : self.truec*vel
        self.noise_vel = lambda pos, vel : self.trued*pos
        self.noise_pos_gen = lambda pos, vel, c: c*vel
        self.noise_vel_gen = lambda pos, vel, d: d*pos

        self.cur_episode = -1

        self.seed() # seed 1 is pretty bad, poor convergence
                    # seed 2 is worse
                    # 5-11 are kinda accurate but late
                    # seed 9 is accurate
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed_saved = seed  # self.np_random.get_state()[1][0]
        return [seed]

    def reset(self):

        self.cur_episode += 1

        # randomly pick the true simulation parameters
        # self.truec = self.np_random.uniform(low=self.TRUECLEFT, high=self.TRUECRIGHT)
        # self.trued = self.np_random.uniform(low=self.TRUEDLEFT, high=self.TRUEDRIGHT)
        self.truec = 0.0
        self.trued = 0.0

        # self.trueinitpos = self.np_random.uniform(low=self.INITPOSLEFT, high=self.INITPOSRIGHT)
        # self.trueinitvel = self.np_random.uniform(low=self.INITVELLEFT, high=self.INITVELRIGHT)
        self.trueinitpos = self.initial_states[self.cur_episode % self.n_initial_states][0]
        self.trueinitvel = self.initial_states[self.cur_episode % self.n_initial_states][1]
        self.flag_state = get_flag_state(self.trueinitpos)

        # self.steepness = self.np_random.choice(self.steepness_vals, 1, p=self.steepness_probs)[0]  # originally 0.0025, challenging = 0.0035
        self.steepness = 0.0025

        # initialize the Gym-related vars
        self.state = np.array([self.trueinitpos, self.trueinitvel])
        self.time = 0

        # Remember the initial observation:
        self.init_pos_obs = self.state[0] + self.noise_pos(self.state[0], self.state[1])
        self.init_vel_obs = self.state[1] + self.noise_vel(self.state[0], self.state[1])

        # Initialize particles
        # self.weights = np.array([1. / self.PARTCT] * self.PARTCT)
        # self.particles_c = self.np_random.uniform(low=self.PFINITCLEFT, high=self.PFINITCRIGHT, size=self.PARTCT)
        # self.particles_d = self.np_random.uniform(low=self.PFINITDLEFT, high=self.PFINITDRIGHT, size=self.PARTCT)
        # self.particles_d = self.np_random.uniform(low=self.trued, high=self.trued, size=self.PARTCT)

        # Create initial GTs from initial observations
        # self.part_pos_gt, self.part_vel_gt = self.obs_to_true(self.init_pos_obs, self.init_vel_obs,self.particles_c, self.particles_d)

        # returning the observation
        return np.array([self.init_pos_obs, self.init_vel_obs, self.flag_state])

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        # proc_noise = self.np_random.normal([0, 0], [self.POSNOISESTD, self.VELNOISESTD])
        proc_noise = np.array([0.0, 0.0])

        # GT state propagation
        position, velocity = self.model_step(position, velocity, force, proc_noise)
        self.state = np.array([position, velocity])

        # updating the simulation state
        done = bool((position >= self.goal_position and velocity >= self.goal_velocity) or self.time > 110)
        reward = 0
        if position >= 0.45:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        # new observed values
        obs_pos = position + self.noise_pos(position, velocity)
        obs_vel = velocity + self.noise_vel(position, velocity)
        self.flag_state = get_flag_state(obs_pos)

        if isinstance(obs_pos, np.ndarray):
            obs_pos = obs_pos[0]

        if isinstance(obs_vel, np.ndarray):
            obs_vel = obs_vel[0]

        obs_state = np.array([obs_pos, obs_vel, self.flag_state])

        self.time += 1

        obs_state = obs_state.reshape((3,))

        return obs_state, reward, done, {}

    def model_step(self, oldpos, oldvel, act, proc_noise):
        # updating state values
        newvel = oldvel + act*self.power - self.steepness * math.cos(3*oldpos) + proc_noise[1]
        if (newvel > self.max_speed): newvel = np.array([self.max_speed])
        if (newvel < -self.max_speed): newvel = np.array([-self.max_speed])
        newpos = oldpos + newvel + proc_noise[0]
        if (newpos > self.max_position): newpos = np.array([self.max_position])
        if (newpos < self.min_position): newpos = np.array([self.min_position])
        if (newpos == self.min_position and newvel<0): newvel = np.array([0])
        return newpos, newvel


# Define custom early stopping callback - stop if the total reward (STL robustness on all bad states) is decreasing
class EarlyStoppingCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.prev_episode_reward = None

    def _on_step(self) -> bool:
        # Check if an episode is done
        if self.locals['done']:
            current_episode_reward = self.locals['infos'][0].get('episode', {}).get('r', 0)

            if self.prev_episode_reward is not None:
                # If the reward of the current episode is smaller than the previous, stop training
                if current_episode_reward < self.prev_episode_reward:
                    if self.verbose > 0:
                        print(
                            f"Stopping training: Current episode reward ({current_episode_reward}) is smaller than previous episode reward ({self.prev_episode_reward}).")
                    return False  # Return False to stop training

            # Update the previous episode reward
            self.prev_episode_reward = current_episode_reward

        return True  # Continue training
