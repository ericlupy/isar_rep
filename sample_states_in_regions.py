import argparse
import os
from incremental_repair_utils import *


def initial_sample(initial_state_regions_path, net_path, sampled_result_path, num_sampled=10, benchmark='uuv'):
    np.random.seed(42)
    df_regions = pd.read_csv(initial_state_regions_path)

    # Load controller network to be repaired
    with open(net_path, 'rb') as f:
        model_dict = yaml.safe_load(f)
    if benchmark == 'uuv':
        net = UUV_Control_NN()
    elif benchmark == 'mc':
        net = MC_Control_NN()
    else:
        raise NotImplementedError
    load_model_dict(model_dict, net)

    if benchmark == 'uuv':
        df_sample = pd.DataFrame(columns=['region', 'y', 'h', 'result'])

        for idx, row in df_regions.iterrows():
            y_lo, y_hi, h_lo, h_hi = row['y_lo'], row['y_hi'], row['h_lo'], row['h_hi']

            print(f'Checking on region {idx}, with {y_lo}, {h_lo} ...')

            for i in range(num_sampled):
                result_dict = {}
                y = np.random.uniform(y_lo, y_hi)
                h = np.random.uniform(h_lo, h_hi)
                _, traj_pos_y = uuv_simulate(net, h, y)
                robustness = uuv_robustness(traj_pos_y)
                result_dict['region'] = idx
                result_dict[f'y'] = y
                result_dict[f'h'] = h
                result_dict[f'result'] = robustness
                df_sample = df_sample.append(result_dict, ignore_index=True)

    elif benchmark == 'mc':
        df_sample = pd.DataFrame(columns=['region', 'pos', 'vel', 'result'])

        for idx, row in df_regions.iterrows():
            pos_lo, pos_hi, vel_lo, vel_hi = row['pos_lo'], row['pos_hi'], row['vel_lo'], row['vel_hi']

            print(f'Checking on region {idx}, with {pos_lo}, {vel_lo} ...')

            for i in range(num_sampled):
                result_dict = {}
                pos = np.random.uniform(pos_lo, pos_hi)
                vel = np.random.uniform(vel_lo, vel_hi)
                traj_pos, _ = mc_simulate(pos, vel, net)
                robustness = mc_robustness(traj_pos)
                result_dict['region'] = idx
                result_dict[f'pos'] = pos
                result_dict[f'vel'] = vel
                result_dict[f'result'] = robustness
                df_sample = df_sample.append(result_dict, ignore_index=True)

    else:
        raise NotImplementedError

    df_sample.to_csv(sampled_result_path)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="uuv or mc", default="uuv")
    parser.add_argument("--network", help="network yml file to be repaired", default=os.path.join('controllers', 'uuv_tanh_2_15_2x32_broken.yml'))
    parser.add_argument("--initial_state_regions_path", help="path to initial state regions csv", default='uuv_initial_state_regions.csv')
    parser.add_argument("--sampled_result_path", help="path to sampling result csv", default='uuv_sampling_result.csv')
    parser.add_argument("--num_samples_per_region", help="number of sampled states in a region", default=10)
    args = parser.parse_args()

    initial_sample(initial_state_regions_path=args.initial_state_regions_path, net_path=args.network,
                   sampled_result_path=args.sampled_result_path, num_sampled=int(args.num_samples_per_region), benchmark=args.benchmark)

