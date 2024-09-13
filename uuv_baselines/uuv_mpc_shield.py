import casadi as ca
import numpy as np
import scipy.io as sio
import math
import os
import pandas as pd
import time
import pickle
import argparse


DEPTH = 45.0
MIN_HEADING_ACTION = math.radians(-5.0)
MAX_HEADING_ACTION = math.radians(5.0)
MIN_SPEED_ACTION = 0.0
MAX_SPEED_ACTION = 1.5433

MIN_SPEED = 0.51444
MAX_SPEED = 2.50
FIXED_SPEED_ACTION = 0.48556

MAX_DISTANCE = 50  # might want to edit these
MIN_DISTANCE = 10

parser = argparse.ArgumentParser()
parser.add_argument("--sampled_result_path", help="path to sampling result csv", default='uuv_sampling_result.csv')
args = parser.parse_args()

# Load model
model = sio.loadmat('model_oneHz.mat')
A, B, C, D = model['A'], model['B'], model['C'], model['D']

# Create non-convex opt problem
opti = ca.Opti()

# Variables
N = 30  # horizon
x = opti.variable(4, N+1)  # State trajectory
u = opti.variable(3, N)  # Control trajectory
y = opti.variable(3, N)

# Costs and dynamics constraints
Q = np.eye(4)  # State cost matrix
R = np.eye(3)  # Control cost matrix
cost = 0
for k in range(N):
    cost += ca.mtimes([x[:, k].T, Q, x[:, k]]) + ca.mtimes([u[:, k].T, R, u[:, k]])
    opti.subject_to(x[:, k+1] == ca.mtimes(A, x[:, k]) + ca.mtimes(B, u[:, k]))
    opti.subject_to(y[:, k] == ca.mtimes(C, x[:, k]) + ca.mtimes(D, u[:, k]))
cost += ca.mtimes([x[:, N-1].T, Q, x[:, N-1]])  # terminal cost

# Initial state constraint
opti.subject_to(x[:, 0] == np.array([0.0, 0.0, 0.0, 0.0]))

# Fixed actions constraints
for k in range(N):
    opti.subject_to(u[1, k] == 0.51444 + 0.48556)
    opti.subject_to(u[2, k] == 45.0)

# Control bound constraints
for k in range(N):
    opti.subject_to(u[0, k] >= math.radians(-5.0))
    opti.subject_to(u[0, k] <= math.radians(5.0))


# Grab failed initial states
print('Obtaining all bad initial states to be repaired ...')
sampled_result_path = args.sampled_result_path
df_sample = pd.read_csv(sampled_result_path)
df_bad = df_sample[df_sample['result'] < 0.0]
bad_states = [(row['y'], row['h']) for index, row in df_bad.iterrows()]

all_bad_states = []
all_optimal_x = []
all_optimal_u = []
all_optimal_y = []

start_time = time.time()

# Solve for all bad states
solved = 0

for bad_state in bad_states:

    print(f'Solving MPC for bad state {bad_state} ...')

    # Add pos_y constraints
    opti_problem = opti.copy()

    init_pos_y = bad_state[0]
    init_heading = math.radians(bad_state[1])
    pos_y = init_pos_y

    for k in range(N):
        heading = y[0, k] + init_heading
        # heading = heading if heading < np.pi else heading - 2 * np.pi
        pos_y -= y[1, k] * ca.sin(heading)
        opti_problem.subject_to(pos_y >= 10.0)
        opti_problem.subject_to(pos_y <= 50.0)

    # Solve for problem
    opti_problem.minimize(cost)
    opti_problem.solver('ipopt')

    try:
        sol = opti_problem.solve()
        print("Optimization problem solved successfully!")
    except Exception as e:
        sol = None
        print("Optimization problem failed:", e)

    if sol is None:
        continue

    x_opt = sol.value(x)
    u_opt = sol.value(u)
    y_opt = sol.value(y)

    solved += 1

    print(f'Solver success, identified x: {x_opt}, u: {u_opt}, y: {y_opt}')

    # Extract the optimal control inputs and states
    all_optimal_x += [x_opt]
    all_optimal_u += [u_opt]
    all_optimal_y += [y_opt]
    all_bad_states += [bad_state]

dict_mpc_data = {'x': all_optimal_x, 'u': all_optimal_u, 'y': all_optimal_y, 'bad_states': all_bad_states}

with open('dict_mpc_data_ipopt_uuv.pkl', 'wb') as f:
    pickle.dump(dict_mpc_data, f)

print(f'Total time = {time.time() - start_time} for {len(bad_states)} of bad initial states, {solved} solved')








