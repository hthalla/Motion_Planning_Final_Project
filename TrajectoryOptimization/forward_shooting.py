"""
This module implements the forward shooting type trajectory optimization.
"""
import os
import numpy as np
import gymnasium as gym
import math
from scipy.optimize import minimize

# Temporary fix to allow simulation while multiple runtime copies
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ForwardShooting():
    """
    Finds the optimized actions to goal position based on Forward Shooting
    trajectory optimization algorithm.
    """
    def __init__(self, env, horizon, start, goal) -> None:
        self.env = env
        self.horizon = horizon
        self.num_actions = 2  # Steering and acceleration
        self.start = start
        self.goal = goal

    def env_reset(self) -> None:
        """
        Resets the env with respect to start and goal position.
        """
        # Resetting complete environment
        self.env.reset()
        # Setting the start position
        self.env.road.vehicles[0].heading = self.start[2]
        self.env.road.vehicles[0].position = np.array([self.start[0],
                                                       self.start[1]])
        # Setting the goal position
        self.env.goal.heading = self.start[2]
        self.env.goal.position = np.array([self.goal[0],
                                           self.goal[1]])

    def forward_shooting(self, actions) -> float:
        """
        Finding the cumulative cost of a given sequences of actions.
        """
        self.env_reset()
        actions = actions.reshape(self.horizon,
                                  self.num_actions)
        total_cost = 0
        for action in actions:
            obs, _, _, _, _ = self.env.step(action)
            theta = math.atan2(obs['observation'][5], obs['observation'][4])
            cost = ((obs['observation'][0] - self.goal[0])**2 +
                    (obs['observation'][1] - self.goal[1])**2 +
                    (theta - self.goal[2])**2)
            total_cost += cost  # Cumulative sum of costs calculation
            # self.env.render()

        return total_cost

    def minimize_shooting(self, init_actions=None) -> list:
        """
        Minimizing the cost function considering the kinematic constraints.
        """
        if init_actions is None:  # Initializing the actions randomly
            init_actions = np.random.uniform(low=0, high=1,
                                             size=(self.horizon *
                                                   self.num_actions,))

        res = minimize(fun=self.forward_shooting,
                       x0=init_actions,
                       method='BFGS',
                       options={'gtol': 1e-4, 'disp': True, 'maxiter': 25})

        return res.x

    def simulate(self, actions) -> None:
        """
        Simulate the environment.
        """
        self.env_reset()
        actions = actions.reshape(self.horizon,
                                  self.num_actions)
        for action in actions:
            self.env.step(action)
            self.env.render()


if __name__ == "__main__":
    park_env = gym.make("parking-v0")
    start_pos = (5.0, 5.0, -np.pi/2)
    goal_pos = (10.0, 1.0, 0)
    H = 5
    traj_shooting = ForwardShooting(park_env, H, start_pos, goal_pos)
    opt_actions = traj_shooting.minimize_shooting()
    traj_shooting.simulate(opt_actions)
