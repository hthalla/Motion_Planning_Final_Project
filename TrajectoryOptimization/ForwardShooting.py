import gymnasium as gym
import numpy as np
import os
from scipy.optimize import minimize

# Temporary fix to allow simulation while multiple runtime copies
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def forward_shooting(actions, env):
    """
    Finding the cumulative cost of a given sequences of actions.
    """
    env.reset(seed=15)
    H = 15  # Horizon
    A = 2  # Actions
    Goal = env.goal.position
    actions = actions.reshape(H, A)

    total_cost = 0

    done = False
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)

        cost = np.sqrt((obs['observation'][0] - Goal[0])**2 +
                       (obs['observation'][1] - Goal[1])**2)
        total_cost += cost  # Cumulative sum of costs calculation
        # env.render()

    return total_cost


def minimize_shooting(env, init_actions=None):
    """
    Minimizing the cost function considering the kinematic constraints.
    """
    if init_actions is None:  # Initializing the actions randomly
        H = 15  # Horizon
        A = 2  # Actions
        init_actions = np.random.uniform(low=0, high=1, size=(H*A,))

    res = minimize(fun=forward_shooting,
                   x0=init_actions,
                   args=env,
                   method='BFGS',
                   options={'gtol': 1e-2, 'disp': True, 'maxiter': 3})

    actions = res.x

    env.reset(seed=15)
    env.render()
    actions = actions.reshape(H, A)

    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    return actions


park_env = gym.make("parking-v0")
traj_shooting = minimize_shooting(park_env)
print(traj_shooting)
