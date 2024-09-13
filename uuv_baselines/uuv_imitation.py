import math
import pickle
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from incremental_repair_utils import *


# Generate training data from pickle file dumped by MPC shield
def generate_data(data_path):

    with open(data_path, 'rb') as file:
        dict_data = pickle.load(file)
    all_optimal_y = dict_data['y']
    all_optimal_u = dict_data['u']
    bad_states = dict_data['bad_states']

    # Check how many of the trajectories meet spec

    all_good_input = []
    all_good_output = []
    count_success = 0

    for i in range(len(bad_states)):
        bad_state = bad_states[i]
        init_pos_y = bad_state[0]
        init_heading = math.radians(bad_state[1])
        pos_y = init_pos_y

        y_opt = all_optimal_y[i]
        pos_ys = []
        headings = []
        stdb_ranges = []
        for k in range(30):
            pos_ys += [pos_y]
            heading = y_opt[0, k] + init_heading
            headings += [-1.0 * heading]
            stdb_ranges += [pos_y / np.cos(heading)]
            pos_y -= y_opt[1, k] * np.sin(heading)

        # MPC has meet spec, append to training data
        if np.max(pos_ys) <= 50.0 and np.min(pos_ys) >= 10.0:
            headings = np.array(headings).reshape(-1, 1)
            stdb_ranges = np.array(stdb_ranges).reshape(-1, 1)
            good_input = np.hstack([headings, stdb_ranges])
            good_output = np.transpose(all_optimal_u[i])[:, 0]
            all_good_input += [good_input]
            all_good_output += [good_output]
            count_success += 1

    all_good_input = np.concatenate(all_good_input)
    all_good_output = np.concatenate(all_good_output)

    print(all_good_input.shape, all_good_output.shape)
    print(f'Among {len(bad_states)} bad states, MPC has solved for {count_success} correctly')

    return all_good_input, all_good_output, bad_states


def imitation(net_path, all_good_input, all_good_output, bad_states, freeze_layers=False, num_epochs=10):

    start_time = time.time()

    net = load_model_dict(net_path)
    net.train()

    inputs = torch.tensor(all_good_input, dtype=torch.float32)
    targets = torch.tensor(all_good_output, dtype=torch.float32)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()  # the same loss as used in MIQP paper
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if freeze_layers:
        for param in net.fc1.parameters():
            param.requires_grad = False
        for param in net.fc2.parameters():
            param.requires_grad = False

    net_prev = net.copy()
    net_prev.eval()

    for epoch in range(num_epochs):

        for batch_inputs, batch_targets in dataloader:
            # Forward pass
            outputs = net(batch_inputs).squeeze()
            loss = criterion(outputs, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check if robustness on bad states are lowered, if so, early stop
        print('Checking if robustness on bad states are lowered ...')
        net.eval()

        h_robustness_bad_prev = []
        h_robustness_bad = []
        for bad_state in bad_states:
            _, traj_y_bad = uuv_simulate(net=net, init_global_heading_deg=bad_state[0], init_pos_y=bad_state[1])
            robustness = uuv_robustness(traj_y_bad)
            _, traj_y_bad_prev = uuv_simulate(net=net_prev, init_global_heading_deg=bad_state[0], init_pos_y=bad_state[1])
            robustness_prev = uuv_robustness(traj_y_bad_prev)
            h_robustness_bad += [robustness]
            h_robustness_bad_prev += [robustness_prev]

        avg_h_robustness_bad_prev = np.mean(h_robustness_bad_prev)
        avg_h_robustness_bad = np.mean(h_robustness_bad)

        print(f'Avg bad states robustness before and after epoch: {avg_h_robustness_bad_prev}, {avg_h_robustness_bad}')

        if avg_h_robustness_bad < avg_h_robustness_bad_prev:
            print('Robustness is lowered during repair, exit')
            break
        else:
            net_prev = net.copy()
            net.train()

    net.eval()
    print(f'Imitation takes {time.time() - start_time} sec')  # 51 sec for not freezing, 37 sec for freezing

    return net


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to pkl file by mpc shield", default='dict_mpc_data_ipopt.pkl')
    parser.add_argument("--network", help="network yml file to be repaired", default=os.path.join('controllers', 'uuv_tanh_2_15_2x32_broken.yml'))
    parser.add_argument("--epochs", help="number of epochs, default = 10", default=10)
    parser.add_argument("--if_miqp", help="if this is MIQP (or minimally deviating repair), default=True", default=True)
    args = parser.parse_args()

    data_path = args.data_path
    all_good_input, all_good_output, bad_states = generate_data(data_path)

    net_path = args.network
    net_repaired = imitation(net_path, all_good_input, all_good_output, bad_states, num_epochs=int(args.epochs), freeze_layers=bool(args.if_miqp))

    if bool(args.if_miqp):
        torch.save(net_repaired, 'repaired_net_miqp.pth')
        dump_model_dict('repaired_net_miqp.yml', net_repaired)
    else:
        torch.save(net_repaired, 'repaired_net_min_deviating.pth')
        dump_model_dict('repaired_net_min_deviating.yml', net_repaired)

