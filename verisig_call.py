import os
import subprocess
import multiprocessing
import argparse
import time
import numpy as np
import pandas as pd


# Parallel function of uuv
def evaluate_conditions_uuv(conditions):
    start_time = time.time()
    test_model = model
    conditions_converted = [conditions[0], conditions[1], conditions[2] * np.pi / 180, conditions[3] * np.pi / 180]
    for i in range(len(legend)):
        test_model = test_model.replace(legend[i], str(conditions_converted[i]))
    with open(output_path + '/uuv_' + str(conditions[0]) + '_' + str(conditions[2]) + '.txt', 'w') as f:
        subprocess.run(flowstar_path + ' ' + controller_yaml_path, input=test_model, shell=True, universal_newlines=True, stdout=f)
    execution_time = time.time() - start_time
    return conditions[0], conditions[2], execution_time


def evaluate_conditions_mc(conditions):
    start_time = time.time()
    test_model = model
    for i in range(len(legend)):
        test_model = test_model.replace(legend[i], str(conditions[i]))
    with open(output_path + '/mc_' + str(conditions[0]) + '_' + str(conditions[2]) + '.txt', 'w') as f:
        subprocess.run(flowstar_path + ' ' + controller_yaml_path, input=test_model, shell=True, universal_newlines=True, stdout=f)
    execution_time = time.time() - start_time
    return conditions[0], conditions[2], execution_time


if __name__ == '__main__':

    # Please run this file after Verisig is installed

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", help="uuv or mc", default="uuv")
    parser.add_argument("--network", help="network yml file to be verified", default=os.path.join('controllers', 'uuv_tanh_2_15_2x32_broken.yml'))
    parser.add_argument("--verisig_path", help="path to verisig directory", default='verisig')
    parser.add_argument("--verisig_output_path", help="path to output txt files of verisig", default=os.path.join('verisig', 'uuv_output'))
    parser.add_argument("--cpu_ratio", help="percentage of cpus in parallel verification, a value between 0 and 1", default=0.5)
    parser.add_argument("--initial_state_regions_path", help="initial state regions csv from previous step", default="uuv_initial_state_regions.csv")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print('Network yaml file not found. Please check your network path.')
        raise FileNotFoundError

    if not os.path.exists(args.verisig_path):
        os.mkdir(args.verisig_path)

    if not os.path.exists(args.verisig_output_path):
        os.mkdir(args.verisig_output_path)

    verisig_path = os.path.join(args.verisig_path, 'verisig')
    flowstar_path = os.path.join(args.verisig_path, 'flowstar', 'flowstar')
    output_path = args.verisig_output_path
    controller_yaml_path = args.network
    controller_name = controller_yaml_path.split('/')[-1][:-4]

    # Decide number of threads in parallel verification - one
    cpu_ratio = float(args.cpu_ratio)
    num_cpus = int(cpu_ratio * multiprocessing.cpu_count())
    if num_cpus > multiprocessing.cpu_count():
        num_cpus = multiprocessing.cpu_count()
    elif num_cpus <= 0:
        num_cpus = 1

    # Load initial state regions
    df_initial_state_regions = pd.read_csv(args.initial_state_regions_path)
    initial_state_regions = df_initial_state_regions.values.tolist()

    if args.benchmark == 'uuv':

        legend = ['Y_LOWER', 'Y_UPPER', 'H_LOWER', 'H_UPPER']

        # This will create a file 'uuv.model'
        print("Building the base model...")
        subprocess.run([verisig_path, '-vc=uuv_multi.yml', '-o', '-nf', 'uuv.xml', controller_yaml_path])

        with open('uuv.model', 'r') as f:
            model = f.read()

        print("Starting parallel verification")
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(evaluate_conditions_uuv, initial_state_regions)
            with open(f'uuv_{controller_name}_execution_log.txt', 'a') as log_file:
                for y_lo, h_lo, execution_time in results:
                    log_file.write(f"{y_lo},{h_lo},{execution_time:.3f}\n")

    elif args.benchmark == 'mc':

        legend = ['X1_LOWER', 'X1_UPPER', 'X2_LOWER', 'X2_UPPER']

        # This will create a file 'mc.model'
        print("Building the base model...")
        subprocess.run([verisig_path, '-vc=mc_multi.yml', '-o', '-nf', 'mc.xml', controller_yaml_path])

        with open('mc.model', 'r') as f:
            model = f.read()

        print("Starting parallel verification")
        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.map(evaluate_conditions_mc, initial_state_regions)
            with open(f'mc_{controller_name}_execution_log.txt', 'a') as log_file:
                for pos_lo, vel_lo, execution_time in results:
                    log_file.write(f"{pos_lo},{vel_lo},{execution_time:.3f}\n")

    else:
        raise NotImplementedError
