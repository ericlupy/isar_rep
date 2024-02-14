import re
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse


def parse(output_dir, x1, x2, benchmark='uuv'):
    print(f'Parsing {x1}, {x2} ...')

    input_filename = os.path.join(output_dir, f'{benchmark}_{x1}_{x2}.txt')

    if not os.path.exists(input_filename):
        return {}

    totalTime = 0
    totalDnnTime = 0
    totalNumBranches = 0
    numRuns = 1

    numSafe = 0
    numUnknown = 0
    numUnsafe = 0
    numNotComp = 0
    final_k = 0

    with open(input_filename, 'r') as f:

        for line in f:

            m = re.search('seconds', line)
            if m is not None:
                items = line.split()
                curTime = items[3]
                curTime = float(curTime[0:len(curTime) - 4])
                totalTime += curTime

            dnn = re.search('dnn', line)
            if dnn is not None:
                items = line.split()
                curTime = float(items[2])
                totalDnnTime += curTime
                # numRuns += 1

            branch = re.search('branches', line)
            if branch is not None:
                items = line.split()
                curBranches = float(items[2])
                totalNumBranches += curBranches

            not_comp = re.search('not completed', line)
            if not_comp is not None:
                numNotComp += 1

            unsafe = re.search('UNSAFE', line)
            if unsafe is not None:
                numUnsafe += 1

            safe = re.search('SAFE', line)
            if safe is not None and unsafe is None:
                numSafe += 1

            unknown = re.search('UNKNOWN', line)
            if unknown is not None:
                numUnknown += 1

            not_supported = re.search('Entered a case that is not supported by Verisig', line)
            if not_supported is not None:
                numUnknown += 1

            k = re.search('^k bounds after reset', line)
            if k is not None:
                items = re.findall("\d+\.\d+", line)
                k_val = items[0]
                k_val = int(float(k_val))
                if k_val > final_k:
                    final_k = k_val

    dict_result = {}
    if numSafe > 0:
        dict_result['result'] = 'safe'
    elif numUnsafe > 0:
        dict_result['result'] = 'unsafe'
    elif numUnknown > 0:
        dict_result['result'] = 'unknown'
    elif numNotComp > 0:
        dict_result['result'] = 'not completed'
    else:
        dict_result['result'] = 'unknown?'
    dict_result['time'] = totalDnnTime

    return dict_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="uuv or mc", default="uuv")
    parser.add_argument("--network", help="network yml file to be verified", default=os.path.join('controllers', 'uuv_tanh_2_15_2x32_broken.yml'))
    parser.add_argument("--verisig_output_path", help="path to output txt files of verisig", default=os.path.join('verisig', 'uuv_output'))
    parser.add_argument("--initial_state_regions_csv", help="initial state regions csv from previous step", default="uuv_initial_state_regions.csv")
    args = parser.parse_args()

    controller_name = args.network.split('/')[-1][:-4]

    # Load initial state regions
    df_initial_state_regions = pd.read_csv(args.initial_state_regions_csv)
    initial_state_regions = df_initial_state_regions.values.tolist()

    output_dir = args.verisig_output_path

    if args.benchmark == 'uuv':
        df_result = pd.DataFrame(columns=['y_lo', 'y_hi', 'h_lo', 'h_hi', 'result'])
        for region in initial_state_regions:
            y_lo, h_lo = region[0], region[2]
            dict_result = parse(output_dir, y_lo, h_lo, benchmark='uuv')
            if len(dict_result.keys()) == 0:
                continue
            dict_row = {'y_lo': region[0], 'y_hi': region[1], 'h_lo': region[2], 'h_hi': region[3], 'result': dict_result['result']}
            df_result = df_result.append(dict_row, ignore_index=True)
        df_result.to_csv('uuv_verisig_result.csv', index=False)

    elif args.benchmark == 'mc':
        df_result = pd.DataFrame(columns=['pos_lo', 'pos_hi', 'vel_lo', 'vel_hi', 'result'])
        for region in initial_state_regions:
            pos_lo, vel_lo = region[0], region[2]
            dict_result = parse(output_dir, pos_lo, vel_lo, benchmark='mc')
            if len(dict_result.keys()) == 0:
                continue
            dict_row = {'pos_lo': region[0], 'pos_hi': region[1], 'vel_lo': region[2], 'vel_hi': region[3], 'result': dict_result['result']}
            df_result = df_result.append(dict_row, ignore_index=True)
        df_result.to_csv('mc_verisig_result.csv', index=False)

    else:
        raise NotImplementedError
