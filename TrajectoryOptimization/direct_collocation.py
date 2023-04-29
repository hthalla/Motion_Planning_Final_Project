"""
This module implements the forward shooting type trajectory optimization.
"""
import os
import numpy as np
import gymnasium as gym
from scipy.optimize import minimize
import math

# Temporary fix to allow simulation while multiple runtime copies
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DirectCollocation():
    """
    Finds the optimized actions to goal position based on Forward Shooting
    trajectory optimization algorithm.
    """
    def __init__(self, env, horizon, markers, start, goal) -> None:
        self.env = env
        self.horizon = horizon
        self.num_actions = 2  # Steering and acceleration
        self.num_states = 3  # x, y and heading
        self.start = start
        self.goal = goal
        self.markers = markers
        self.bounds = (((0, 25), (0, 25), (-np.pi, np.pi))*self.horizon +
                       ((0, 1), (-1, 1))*self.horizon)  # Dubins car model

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

    def env_set_state(self, state) -> None:
        """
        Sets the state of the vehicle in the desired position.
        """
        self.env.road.vehicles[0].heading = state[2]
        self.env.road.vehicles[0].position = np.array([state[0],
                                                       state[1]])

    def direct_collocation(self, states_actions) -> float:
        """
        Finding the cumulative cost of a given sequences of actions.
        """
        self.env_reset()
        (states, actions) = (states_actions[:self.horizon * self.num_states],
                             states_actions[self.horizon * self.num_states:])
        states = states.reshape(self.horizon, self.num_states)
        actions = actions.reshape(self.horizon, self.num_actions)
        total_cost = 0
        for action in actions:
            obs, _, done, _, _ = self.env.step(action)
            theta = math.atan2(obs['observation'][5], obs['observation'][4])
            cost = ((obs['observation'][0] - self.goal[0])**2 +
                    (obs['observation'][1] - self.goal[1])**2 +
                    (theta - self.goal[2])**2)
            total_cost += cost  # Cumulative sum of costs calculation
            # self.env.render()
        print(done, total_cost)

        return total_cost

    def minimize_collocation(self, init_states_actions=None) -> list:
        """
        Minimizing the cost function considering the kinematic constraints.
        """
        if init_states_actions is None:  # Random initialization
            init_states_actions = np.zeros(shape=(self.horizon *
                                                  (self.num_states +
                                                   self.num_actions),))

            # Initializing the start location
            init_states_actions[:3] = np.array(list(self.start))

            # Initializing x
            init_states_actions[3:self.horizon*self.num_states][::3] = \
                np.random.uniform(low=self.start[0]-5, high=self.start[0]+5,
                                  size=((self.horizon-1),))

            # Initializing y
            init_states_actions[3:self.horizon*self.num_states][1::3] = \
                np.random.uniform(low=self.start[1]-5, high=self.start[1]+5,
                                  size=((self.horizon-1),))

            # Initializing theta
            init_states_actions[3:self.horizon*self.num_states][2::3] = \
                np.random.uniform(low=-np.pi, high=np.pi,
                                  size=((self.horizon-1),))

            # Initializing acceleration
            init_states_actions[self.horizon*self.num_states:][::2] = \
                np.random.uniform(low=0, high=1, size=(self.horizon,))

            # Initializing steering
            init_states_actions[self.horizon*self.num_states:][1::2] = \
                np.random.uniform(low=-1, high=1, size=(self.horizon,))

        eq_cons = {'type': 'eq',
                   'fun': lambda x: self.set_constraints(x)}
        res = minimize(fun=self.direct_collocation,
                       x0=init_states_actions,
                       method='SLSQP',
                       constraints=eq_cons,
                       bounds=self.bounds,
                       options={'ftol': 1e-3, 'disp': True, 'maxiter': 10})

        return res.x

    def set_constraints(self, states_actions):
        """
        Setting the constraints.
        """
        self.env_reset()
        constraints = []
        (states, actions) = (states_actions[:self.horizon * self.num_states],
                             states_actions[self.horizon * self.num_states:])
        states = states.reshape(self.horizon, self.num_states)
        actions = actions.reshape(self.horizon, self.num_actions)

        self.env_set_state(states[0])  # Setting the env with initial state
        for state, action in zip(states[1:self.horizon],
                                 actions[:(self.horizon-1)]):
            obs, _, _, _, _ = self.env.step(action)
            theta = math.atan2(obs['observation'][5], obs['observation'][4])
            constraints += [state[0] - obs["observation"][0],
                            state[1] - obs["observation"][1],
                            state[2] - theta]
            self.env_set_state(state)  # Setting state for next iter

        return np.array(constraints)

    def simulate(self, states_actions) -> None:
        """
        Simulate the environment.
        """
        self.env_reset()
        (states, actions) = (states_actions[:self.horizon * self.num_states],
                             states_actions[self.horizon * self.num_states:])
        states = states.reshape(self.horizon, self.num_states)
        actions = actions.reshape(self.horizon, self.num_actions)
        for action in actions:
            self.env.step(action)
            self.env.render()


if __name__ == "__main__":
    park_env = gym.make("parking-v0")
    start_pos = (5.0, 5.0, -np.pi/2)
    goal_pos = (10.0, 1.0, 0)
    land_marks = [(10.0, 5.0, 0.0), (15.0, 3.0, 0.0)]  # For future use
    H = 5
    traj_collocation = DirectCollocation(park_env, H, land_marks,
                                         start_pos, goal_pos)
    opt_states_actions = traj_collocation.minimize_collocation()
    traj_collocation.simulate(opt_states_actions)
