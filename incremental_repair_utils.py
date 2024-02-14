import numpy as np
import torch
import torch.nn as nn
import yaml
import scipy.io as sio
import pandas as pd

""" Controller network utils """


# PyTorch models for UUV controller net
class UUV_Control_NN(nn.Module):

    def __init__(self, layer_1_size=32, layer_2_size=32):
        super(UUV_Control_NN, self).__init__()
        self.fc1 = nn.Linear(2, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# PyTorch model for mc controller net
class MC_Control_NN(nn.Module):

    def __init__(self, layer_1_size=16, layer_2_size=16):
        super(MC_Control_NN, self).__init__()
        self.fc1 = nn.Linear(2, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


# Load yaml to torch model
def load_model_dict(model_dict: dict, network: nn.Module):
    state_dict = {}
    state_dict['fc1.weight'] = torch.FloatTensor(model_dict['weights'][1])
    state_dict['fc1.bias'] = torch.FloatTensor(model_dict['offsets'][1])
    state_dict['fc2.weight'] = torch.FloatTensor(model_dict['weights'][2])
    state_dict['fc2.bias'] = torch.FloatTensor(model_dict['offsets'][2])
    state_dict['fc3.weight'] = torch.FloatTensor(model_dict['weights'][3])
    state_dict['fc3.bias'] = torch.FloatTensor(model_dict['offsets'][3])
    network.load_state_dict(state_dict)


# Dump torch model to yaml
def dump_model_dict(filename, network: nn.Module):
    model_dict = {}
    model_dict['activations'] = {}
    if isinstance(network, UUV_Control_NN):
        model_dict['activations'][1] = 'Tanh'
        model_dict['activations'][2] = 'Tanh'
    elif isinstance(network, MC_Control_NN):
        model_dict['activations'][1] = 'Sigmoid'
        model_dict['activations'][2] = 'Sigmoid'
    else:
        raise NotImplementedError
    model_dict['activations'][3] = 'Tanh'
    model_dict['weights'] = {}
    model_dict['offsets'] = {}
    for layer in [1, 2, 3]:
        model_dict['weights'][layer] = network.state_dict()[f'fc{layer}.weight'].tolist()
        model_dict['offsets'][layer] = network.state_dict()[f'fc{layer}.bias'].tolist()
    with open(filename, 'w') as f:
        yaml.dump(model_dict, f)
    return


""" UUV utils """


# Simulate, init_global_heading is in degrees
def uuv_simulate(net: UUV_Control_NN, init_global_heading_deg, init_pos_y):
    # Dynamics coeff matrices
    coeffs = sio.loadmat('uuv_model_oneHz.mat')
    A, B, C, D = coeffs['A'], coeffs['B'], coeffs['C'], coeffs['D']

    init_global_heading = init_global_heading_deg / 180 * np.pi
    x = np.array([[0], [0], [0], [0]])
    u = np.array([[0], [0.48556], [45.0]])
    pos_x = 0.0
    pos_y = init_pos_y

    traj_pos_x = [pos_x]
    traj_pos_y = [pos_y]

    for i in range(30):

        # Compute y and x
        y = np.dot(C, x) + np.dot(D, u)
        x = np.dot(A, x) + np.dot(B, u)

        # Update pos_x, pos_y
        heading = y[0][0]
        heading = heading if heading < np.pi else heading - 2 * np.pi
        global_heading = heading + init_global_heading
        pos_x += y[1][0] * np.cos(global_heading)
        pos_y -= y[1][0] * np.sin(global_heading)
        traj_pos_x += [pos_x]
        traj_pos_y += [pos_y]

        # Early stop
        if pos_y < 10 or pos_y > 50 or pos_x < -10 or pos_x > 400:
            break

        # Control u update
        pipe_heading = -1.0 * global_heading
        stdb_range = pos_y / np.cos(global_heading)
        nn_inputs = torch.FloatTensor([pipe_heading, stdb_range])
        nn_out = net(nn_inputs).detach().numpy()[0]
        heading_delta = np.radians(5) * nn_out
        abs_heading = heading_delta + heading
        abs_heading = abs_heading if abs_heading < np.pi else abs_heading - 2 * np.pi
        u = np.array([[abs_heading], [0.48556], [45.0]])

    return traj_pos_x, traj_pos_y


# STL robustness of G_{[0, 30]} (pos_y \leq 50 \land pos_y \geq 10)
# for each pos_y, find min(50 - pos_y, pos_y - 10)
def uuv_robustness(traj_pos_y):
    scores = [min(50 - pos_y, pos_y - 10) for pos_y in traj_pos_y]
    return np.min(scores)


# Objective function of UUV
def uuv_barriered_energy(bad_states, good_states, net, lambda_=1.0):
    h_robustness_bad = []
    for bad_state in bad_states:
        _, traj_y_bad = uuv_simulate(net=net, init_pos_y=bad_state[0], init_global_heading_deg=bad_state[1])
        h_robustness_bad += [uuv_robustness(traj_y_bad)]

    log_barriers = []
    h_robustness_good = []
    for good_state in good_states:
        _, traj_y_good = uuv_simulate(net=net, init_pos_y=good_state[0], init_global_heading_deg=good_state[1])
        robustness = uuv_robustness(traj_y_good)
        h_robustness_good += [robustness]
        if robustness > 0.0 and np.log(robustness) > -1000:
            log_barrier = np.log(robustness)
        else:
            log_barrier = -1000
        log_barriers += [log_barrier]

    return lambda_ * np.mean(h_robustness_bad) + np.mean(log_barriers), h_robustness_bad, h_robustness_good


""" MC utils """


# Simulate traj_pos and traj_vel from an initial state (replace this with actual simulator later)
def mc_simulate(pos_0, vel_0, net: MC_Control_NN, length=111, steepness=0.0025):
    net.eval()
    traj_pos = np.zeros(length)
    traj_vel = np.zeros(length)
    traj_pos[0] = pos_0
    traj_vel[0] = vel_0
    for i in range(1, length):
        traj_pos[i] = traj_pos[i - 1] + traj_vel[i - 1]
        inputs = torch.FloatTensor([traj_pos[i - 1], traj_vel[i - 1]])
        u = net(inputs).detach().numpy()
        traj_vel[i] = traj_vel[i - 1] + 0.0015 * u - steepness * np.cos(3 * traj_pos[i - 1])
        # simulator constraints
        if (traj_vel[i] > 0.07): traj_vel[i] = 0.07
        if (traj_vel[i] < -0.07): traj_vel[i] = -0.07
        if (traj_pos[i] > 0.6): traj_pos[i] = 0.6
        if (traj_pos[i] < -1.2): traj_pos[i] = -1.2
        if (traj_pos[i] == -1.2 and traj_vel[i] < 0): traj_vel[i] = 0
    return traj_pos, traj_vel


# STL robustness of MC
def mc_robustness(traj_pos):
    return np.max(traj_pos) - 0.45


# Objective function
def mc_barriered_energy(bad_states, good_states, net, lambda_=1.0):
    h_robustness_bad = []
    for bad_state in bad_states:
        traj_pos_bad, _ = mc_simulate(bad_state[0], bad_state[1], net)
        h_robustness_bad += [mc_robustness(traj_pos_bad)]

    log_barriers = []
    h_robustness_good = []
    for good_state in good_states:
        traj_pos_good, _ = mc_simulate(good_state[0], good_state[1], net)
        robustness = mc_robustness(traj_pos_good)
        h_robustness_good += [robustness]
        if robustness > 0.0 and np.log(robustness) > -1000:
            log_barrier = np.log(robustness)
        else:
            log_barrier = -1000
        log_barriers += [log_barrier]

    return lambda_ * np.mean(h_robustness_bad) + np.mean(log_barriers), h_robustness_bad, h_robustness_good


""" Other utils """


# Color the regions into red, yellow, green
# Red: counterexample sampled; yellow: no counterexample sampled, but verification fails; green: verification passes
def color_regions(verisig_result_path, sampled_result_path):
    df_verisig = pd.read_csv(verisig_result_path)
    df_sample = pd.read_csv(sampled_result_path)

    dict_color = {}
    count_green = 0
    count_yellow = 0
    count_red = 0

    for idx, row in df_verisig.iterrows():
        verisig_result = row['result']  # safe, unknown, unsafe
        df_region_sample = df_sample[df_sample['region'] == idx]
        sampled_result = (df_region_sample['result'] >= 0).all()  # true if all safe, false otherwise
        if verisig_result == 'safe' or verisig_result == 'unknown?':
            dict_color[idx] = 'green'
            count_green += 1
        elif sampled_result:
            dict_color[idx] = 'yellow'
            count_yellow += 1
        else:
            dict_color[idx] = 'red'
            count_red += 1
    print(f'Green, green hatches, red: {count_green}, {count_yellow}, {count_red}')
    return dict_color


# Sort a set of regions in decreasing order of average robustness
def sort_regions(df, benchmark='uuv'):

    if df.empty:
        return []
    region_robustness = []

    if benchmark == 'uuv':
        num_regions = 2000
    elif benchmark == 'mc':
        num_regions = 900
    else:
        raise NotImplementedError

    for i in range(num_regions):
        df_region = df[df['region'] == i]
        if df_region.empty:
            continue
        avg_robustness = df_region['result'].mean()
        region_robustness += [[i, avg_robustness]]

    # Sort by robustness in decreasing order
    region_robustness.sort(key=lambda x: x[1], reverse=True)

    return region_robustness


# Check robustness on sampled states
def check_samples(repaired_net_path, sampled_result_path, sample_repaired_result_path, benchmark='uuv'):

    if repaired_net_path.endswith('.yml'):
        with open(repaired_net_path, 'rb') as f:
            model_dict = yaml.safe_load(f)
        if benchmark == 'uuv':
            net = UUV_Control_NN()
        elif benchmark == 'mc':
            net = MC_Control_NN()
        else:
            raise NotImplementedError
        load_model_dict(model_dict, net)
    else:
        net = torch.load(repaired_net_path)
    df_sample = pd.read_csv(sampled_result_path)
    print('Checking sampled states ...')

    if benchmark == 'uuv':
        df_sample_repaired = pd.DataFrame(columns=['region', 'y', 'h', 'result'])
        for idx, row in df_sample.iterrows():
            result_dict = {}
            region, y, h = row['region'], row['y'], row['h']
            _, traj_pos_y = uuv_simulate(net=net, init_global_heading_deg=h, init_pos_y=y)
            robustness = uuv_robustness(traj_pos_y)
            result_dict['region'] = region
            result_dict['y'] = y
            result_dict['h'] = h
            result_dict['result'] = robustness
            df_sample_repaired = df_sample_repaired.append(result_dict, ignore_index=True)
    elif benchmark == 'mc':
        df_sample_repaired = pd.DataFrame(columns=['region', 'pos', 'vel', 'result'])
        for idx, row in df_sample.iterrows():
            result_dict = {}
            region, pos, vel = row['region'], row['pos'], row['vel']
            traj_pos, _ = mc_simulate(pos, vel, net)
            robustness = mc_robustness(traj_pos)
            result_dict['region'] = region
            result_dict[f'pos'] = pos
            result_dict[f'vel'] = vel
            result_dict[f'result'] = robustness
            df_sample_repaired = df_sample_repaired.append(result_dict, ignore_index=True)
    else:
        raise NotImplementedError

    df_sample_repaired.to_csv(sample_repaired_result_path, index=False)
    return
