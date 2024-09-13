import casadi as ca
import numpy as np
import os
import pandas as pd
import time
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sampled_result_path", help="path to sampling result csv", default='uuv_sampling_result.csv')
args = parser.parse_args()


# Create non-convex opt problem
opti = ca.Opti()

# Variables
N = 110  # horizon
x = opti.variable(2, N+1)  # State trajectory
u = opti.variable(1, N)  # Control trajectory

# Costs and dynamics constraints
Q = np.eye(2)  # State cost matrix
R = np.eye(1)  # Control cost matrix
cost = 0

for k in range(N):
    cost += ca.mtimes([x[:, k].T, Q, x[:, k]]) + ca.mtimes([u[:, k].T, R, u[:, k]])
    opti.subject_to(x[1, k+1] == x[1, k] + 0.0015 * u[0, k] - 0.0025 * ca.cos(3 * x[0, k]))
    opti.subject_to(x[0, k+1] == x[0, k] + x[1, k+1])

cost += ca.mtimes([x[:, N-1].T, Q, x[:, N-1]])  # terminal cost
opti.subject_to(x[0, N-1] >= 0.45)  # target

# Control and state bound constraints
for k in range(N):
    opti.subject_to(u[0, k] >= -1.0)
    opti.subject_to(u[0, k] <= 1.0)
    opti.subject_to(x[0, k] >= -1.2)
    opti.subject_to(x[0, k] <= 0.6)
    opti.subject_to(x[1, k] >= -0.07)
    opti.subject_to(x[1, k] <= 0.07)


# Grab failed initial states
print('Obtaining all bad initial states to be repaired ...')
sampled_result_path = args.sampled_result_path
df_sample = pd.read_csv(sampled_result_path)
df_bad = df_sample[df_sample['result'] < 0.0]
bad_states = [(row['pos'], row['vel']) for index, row in df_bad.iterrows()]

all_bad_states = []
all_optimal_x = []
all_optimal_u = []

start_time = time.time()

# Solve for all bad states
solved = 0

for bad_state in bad_states:

    print(f'Solving MPC for bad state {bad_state} ...')

    # Add pos_y constraints
    opti_problem = opti.copy()

    init_pos, init_vel = bad_state[0], bad_state[1]

    # Initial state constraint
    opti.subject_to(x[:, 0] == np.array([init_pos, init_vel]))

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

    solved += 1

    print(f'Solver success, identified x: {x_opt}, u: {u_opt}')

    # Extract the optimal control inputs and states
    all_optimal_x += [x_opt]
    all_optimal_u += [u_opt]
    all_bad_states += [bad_state]

dict_mpc_data = {'x': all_optimal_x, 'u': all_optimal_u, 'bad_states': all_bad_states}

with open(f'dict_mpc_data_ipopt_mc.pkl', 'wb') as f:
    pickle.dump(dict_mpc_data, f)

print(f'Total time = {time.time() - start_time} for {len(bad_states)} of bad initial states, {solved} solved')








