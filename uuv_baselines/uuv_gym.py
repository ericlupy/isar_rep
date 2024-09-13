import gym
from gym import spaces
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
from stable_baselines3.common.callbacks import BaseCallback


# car parameters
DEPTH = 45.0
MIN_HEADING_ACTION = math.radians(-5.0)
MAX_HEADING_ACTION = math.radians(5.0)
MIN_SPEED_ACTION = 0.0
MAX_SPEED_ACTION = 1.5433

MIN_SPEED = 0.51444
MAX_SPEED = 2.50
FIXED_SPEED_ACTION = 0.48556

# training parameters
STEP_REWARD_GAIN = 0.5
HEADING_REWARD_GAIN = 5
INPUT_REWARD_GAIN = -0.5
RANGE_REWARD_PENALTY = -0.1
CRASH_PENALTY = -100

GOAL_RANGE = 30.0

MAX_DISTANCE = 50  # might want to edit these
MIN_DISTANCE = 10

PIPE_LENGTH = 400

INIT_HEADING_RANGE = math.radians(30)  # 0.06

STARTING_DISTANCE = 30
DISTANCE_RANGE = 10  # 2


class UUVEnv(gym.Env):

    def __init__(self, initial_states, episode_length=30):

        self.initial_states = initial_states  # bad states to be repaired by retraining
        self.n_initial_states = len(initial_states)

        self.pos_x = 0.0
        self.init_pos_y = 0.0
        self.pos_y = self.init_pos_y
        self.init_heading = 0.0
        self.init_global_heading = self.init_heading
        self.heading = self.init_heading

        model = sio.loadmat('model_oneHz.mat')

        self.A = model['A']
        self.B = model['B']
        self.C = model['C']
        self.D = model['D']
        self.K = model['K']

        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.u = np.array([[0], [FIXED_SPEED_ACTION], [DEPTH]])

        # step parameters
        self.cur_step = 0
        self.cur_episode = -1
        self.episode_length = episode_length

        # storage
        self.allX = []
        self.allY = []
        self.allH = []
        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allH.append(self.init_global_heading)

        # parameters needed for consistency with gym environments
        self.obs_low = np.array([math.radians(-180.0), MIN_DISTANCE])
        self.obs_high = np.array([math.radians(180.0), MAX_DISTANCE])

        self.action_space = spaces.Box(low=MIN_HEADING_ACTION, high=MAX_HEADING_ACTION, shape=(1,))
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

        self.max_episode_steps = episode_length
        self.reward_cache = []

    def reset(self):
        self.cur_episode += 1
        self.cur_step = 0
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.u = np.array([[0], [0.48556], [45.0]])

        init_pos_y = self.initial_states[self.cur_episode % self.n_initial_states][0]
        init_global_heading_deg = self.initial_states[self.cur_episode % self.n_initial_states][1]

        self.pos_x = 0.0
        self.pos_y = init_pos_y
        self.heading = init_global_heading_deg / 180 * np.pi
        self.init_global_heading = self.heading

        self.allX = []
        self.allY = []
        self.allH = []
        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allH.append(self.init_global_heading)
        self.reward_cache = []

        pipe_heading = -1.0 * self.init_global_heading  # HEADING_NOISE * (np.random.random() - 0.5)
        stbd_range = self.pos_y / math.cos(self.init_global_heading)  # RANGE_NOISE * (np.random.random() - 0.5)
        measurements = np.array([pipe_heading, stbd_range])
        return measurements

    def step(self, action):
        self.cur_step += 1

        heading_delta = action[0]
        speed = FIXED_SPEED_ACTION

        # Constrain turning input
        if heading_delta > MAX_HEADING_ACTION:
            heading_delta = MAX_HEADING_ACTION

        if heading_delta < MIN_HEADING_ACTION:
            heading_delta = MIN_HEADING_ACTION

        if speed < MIN_SPEED_ACTION:
            speed = MIN_SPEED_ACTION

        if speed > MAX_SPEED_ACTION:
            speed = MAX_SPEED_ACTION

        abs_heading = self.heading + heading_delta
        abs_heading = abs_heading if abs_heading < math.pi else abs_heading - (2 * math.pi)

        # print(abs_heading)

        u = np.array([[abs_heading], [MIN_SPEED + speed], [45.0]])

        y = np.dot(self.C, self.x) + np.dot(self.D, u)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)

        self.heading = y[0][0]
        self.heading = self.heading if self.heading < math.pi else self.heading - (2 * math.pi)

        global_heading = self.heading + self.init_global_heading
        self.pos_x += y[1][0] * math.cos(global_heading)
        self.pos_y += y[1][0] * -1.0 * math.sin(global_heading)

        terminal = False

        # Early stopping conditions
        if self.pos_x > PIPE_LENGTH:  # pos_x goes off end of pipe
            # print("off the end")
            terminal = True
        elif self.pos_x < -10.0:  # pos_x goes off beginning of pipe
            # print("off the beginning")
            terminal = True

        if self.pos_y > MAX_DISTANCE or self.pos_y < MIN_DISTANCE:  # pos_y out of safe zone
            # print("too far")
            terminal = True

        if self.cur_step == self.episode_length:
            terminal = True

        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allH.append(global_heading)

        # Measurements
        pipe_heading = -1.0 * global_heading # HEADING_NOISE * (np.random.random() - 0.5)
        stbd_range = self.pos_y / math.cos(global_heading)  # RANGE_NOISE * (np.random.random() - 0.5)
        measurements = np.array([pipe_heading, stbd_range])

        # Compute reward
        dist = min(MAX_DISTANCE - self.pos_y, self.pos_y - MIN_DISTANCE)
        self.reward_cache.append(-dist)
        if len(self.reward_cache) < 10:
            reward = max(self.reward_cache)
        else:
            reward = max(self.reward_cache[-10:])

        return measurements, reward, terminal, {}

    def plot_trajectory(self):
        fig = plt.figure()
        plt.plot(np.array([0.0, PIPE_LENGTH]), np.array([0.0, 0.0]), 'b', linewidth=3)
        plt.plot(self.allX, self.allY, 'r--')
        plt.show()


# FMDP flag states:
def get_flag_state(y, t):
    if (10 < y <= 13 or 47 <= y < 50) and (0 <= t < 5):
        return 1
    elif (10 < y <= 13 or 47 <= y < 50) and (5 <= t < 10):
        return 2
    elif (10 < y <= 13 or 47 <= y < 50) and (10 <= t < 15):
        return 3
    elif (10 < y <= 13 or 47 <= y < 50) and (15 <= t < 20):
        return 4
    elif (10 < y <= 13 or 47 <= y < 50) and (20 <= t < 25):
        return 5
    elif (10 < y <= 13 or 47 <= y < 50) and (25 <= t < 30):
        return 6
    else:
        return 0


class UUVEnvFMDP(gym.Env):

    def __init__(self, initial_states, episode_length=30):

        self.initial_states = initial_states  # bad states to be repaired by retraining
        self.n_initial_states = len(initial_states)

        self.pos_x = 0.0
        self.init_pos_y = 0.0
        self.pos_y = self.init_pos_y
        self.init_heading = 0.0
        self.init_global_heading = self.init_heading
        self.heading = self.init_heading
        self.flag_state = 0

        model = sio.loadmat('model_oneHz.mat')

        self.A = model['A']
        self.B = model['B']
        self.C = model['C']
        self.D = model['D']
        self.K = model['K']

        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.u = np.array([[0], [FIXED_SPEED_ACTION], [DEPTH]])

        # step parameters
        self.cur_step = 0
        self.cur_episode = -1
        self.episode_length = episode_length

        # storage
        self.allX = []
        self.allY = []
        self.allH = []
        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allH.append(self.init_global_heading)

        # parameters needed for consistency with gym environments
        # observation space adds one flag state dimension
        self.obs_low = np.array([math.radians(-180.0), MIN_DISTANCE, 0])
        self.obs_high = np.array([math.radians(180.0), MAX_DISTANCE, 6])

        self.action_space = spaces.Box(low=MIN_HEADING_ACTION, high=MAX_HEADING_ACTION, shape=(1,))
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

        self.max_episode_steps = episode_length
        self.reward_cache = []

    def reset(self):
        self.cur_episode += 1
        self.cur_step = 0
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.u = np.array([[0], [0.48556], [45.0]])

        init_pos_y = self.initial_states[self.cur_episode % self.n_initial_states][0]
        init_global_heading_deg = self.initial_states[self.cur_episode % self.n_initial_states][1]
        self.flag_state = get_flag_state(init_pos_y, 0)

        self.pos_x = 0.0
        self.pos_y = init_pos_y
        self.heading = init_global_heading_deg / 180 * np.pi
        self.init_global_heading = self.heading

        self.allX = []
        self.allY = []
        self.allH = []
        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allH.append(self.init_global_heading)
        self.reward_cache = []

        pipe_heading = -1.0 * self.init_global_heading  # HEADING_NOISE * (np.random.random() - 0.5)
        stbd_range = self.pos_y / math.cos(self.init_global_heading)  # RANGE_NOISE * (np.random.random() - 0.5)
        measurements = np.array([pipe_heading, stbd_range, self.flag_state])  # one additional flag state
        return measurements

    def step(self, action):
        self.cur_step += 1

        heading_delta = action[0]
        speed = FIXED_SPEED_ACTION

        # Constrain turning input
        if heading_delta > MAX_HEADING_ACTION:
            heading_delta = MAX_HEADING_ACTION

        if heading_delta < MIN_HEADING_ACTION:
            heading_delta = MIN_HEADING_ACTION

        if speed < MIN_SPEED_ACTION:
            speed = MIN_SPEED_ACTION

        if speed > MAX_SPEED_ACTION:
            speed = MAX_SPEED_ACTION

        abs_heading = self.heading + heading_delta
        abs_heading = abs_heading if abs_heading < math.pi else abs_heading - (2 * math.pi)

        # print(abs_heading)

        u = np.array([[abs_heading], [MIN_SPEED + speed], [45.0]])

        y = np.dot(self.C, self.x) + np.dot(self.D, u)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)

        self.heading = y[0][0]
        self.heading = self.heading if self.heading < math.pi else self.heading - (2 * math.pi)

        global_heading = self.heading + self.init_global_heading
        self.pos_x += y[1][0] * math.cos(global_heading)
        self.pos_y += y[1][0] * -1.0 * math.sin(global_heading)

        self.flag_state = get_flag_state(self.pos_y, self.cur_step)

        terminal = False

        # Early stopping conditions
        if self.pos_x > PIPE_LENGTH:  # pos_x goes off end of pipe
            # print("off the end")
            terminal = True
        elif self.pos_x < -10.0:  # pos_x goes off beginning of pipe
            # print("off the beginning")
            terminal = True

        if self.pos_y > MAX_DISTANCE or self.pos_y < MIN_DISTANCE:  # pos_y out of safe zone
            # print("too far")
            terminal = True

        if self.cur_step == self.episode_length:
            terminal = True

        self.allX.append(self.pos_x)
        self.allY.append(self.pos_y)
        self.allH.append(global_heading)

        # Measurements
        pipe_heading = -1.0 * global_heading # HEADING_NOISE * (np.random.random() - 0.5)
        stbd_range = self.pos_y / math.cos(global_heading)  # RANGE_NOISE * (np.random.random() - 0.5)
        measurements = np.array([pipe_heading, stbd_range, self.flag_state])

        # Compute reward
        dist = min(MAX_DISTANCE - self.pos_y, self.pos_y - MIN_DISTANCE)
        self.reward_cache.append(-dist)
        if len(self.reward_cache) < 10:
            reward = max(self.reward_cache)
        else:
            reward = max(self.reward_cache[-10:])

        return measurements, reward, terminal, {}

    def plot_trajectory(self):
        fig = plt.figure()
        plt.plot(np.array([0.0, PIPE_LENGTH]), np.array([0.0, 0.0]), 'b', linewidth=3)
        plt.plot(self.allX, self.allY, 'r--')
        plt.show()


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
