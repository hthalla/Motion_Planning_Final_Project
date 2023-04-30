"""
This module implements the forward shooting type trajectory optimization.
"""
import cv2
import numpy as np
import gymnasium as gym
from scipy.optimize import minimize


class DirectCollocation():
    """
    Finds the optimized actions to goal position based on Forward Shooting
    trajectory optimization algorithm.
    """
    def __init__(self, env, horizon, start, goal) -> None:
        self.env = env
        self.horizon = horizon
        self.num_actions = 2  # Steering and acceleration
        self.num_states = 3  # x, y and heading
        self.start = start
        self.goal = goal
        self.images = []
        self.bounds = (((-31, 31), (-19, 19), (-np.pi, np.pi))*self.horizon +
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
        self.env.goal.heading = self.goal[2]
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
            self.env.step(action)
            cur_pos = self.env.road.vehicles[0]
            cost = ((cur_pos.position[0] - self.goal[0])**2 +
                    (cur_pos.position[1] - self.goal[1])**2 +
                    (cur_pos.heading - self.goal[2])**2)
            total_cost += cost  # Cumulative sum of costs calculation

        return total_cost

    def minimize_collocation(self, init_states_actions=None) -> list:
        """
        Minimizing the cost function considering the kinematic constraints.
        """
        if init_states_actions is None:  # Random initialization
            init_states_actions = np.zeros(shape=(self.horizon *
                                                  (self.num_states +
                                                   self.num_actions),))

            # Initializing x
            init_states_actions[:self.horizon*self.num_states][::3] = \
                np.linspace(self.start[0], self.goal[0], self.horizon)

            # Initializing y
            init_states_actions[:self.horizon*self.num_states][1::3] = \
                np.linspace(self.start[1], self.goal[1], self.horizon)

            # Initializing theta
            init_states_actions[:self.horizon*self.num_states][2::3] = \
                np.linspace(self.start[2], self.goal[2], self.horizon)

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
            self.env.step(action)
            cur_pos = self.env.road.vehicles[0]
            constraints += [state[0] - cur_pos.position[0],
                            state[1] - cur_pos.position[1],
                            state[2] - cur_pos.heading]
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
        img = self.env.render()
        self.images += [img]


if __name__ == "__main__":
    park_env = gym.make("parking-v0")

    # Start position
    start_pos = (-30.5, 17.5, -1.4959965017094252)
    H = 5

    # Sample trajectory
    land_marks = [(-30.5, 17.5, -1.4959965017094252),
                  (-30.35053981, 15.505592, -1.495996),
                  (-30.87578469, 13.617033, -2.188816),
                  (-31.40233787, 11.728838, -1.495996),
                  (-30.60148439, 9.9396562, -0.803176),
                  (-28.84250145, 9.0744872, -0.110355),
                  (-26.85466751, 8.8542232, -0.110355),
                  (-24.86683357, 8.6339592, -0.110355),
                  (-22.87899963, 8.4136952, -0.110355),
                  (-20.89116569, 8.1934312, -0.110355),
                  (-18.90333175, 7.9731672, -0.110355),
                  (-16.91549781, 7.7529033, -0.110355),
                  (-14.92766387, 7.5326393, -0.110355),
                  (-12.93982993, 7.3123753, -0.110355),
                  (-10.95199599, 7.0921113, -0.110355),
                  (-8.964162060, 6.8718473, -0.110355),
                  (-6.976328120, 6.6515833, -0.110355),
                  (-4.988494180, 6.4313193, -0.110355),
                  (-3.000660241, 6.2110553, -0.110355),
                  (-1.241078319, 5.3471052, -0.803176),
                  (0.5179046181, 4.4819362, -0.110355),
                  (2.4239419219, 4.9397136, 0.5824644),
                  (2.4239419219, 4.9397136, 0.5824644)]

    traj_collocation = DirectCollocation(park_env, H,
                                         start_pos, land_marks[0])

    opt_states = np.array([])
    opt_actions = np.array([])
    i = 0
    while i < len(land_marks):
        traj_collocation.goal = land_marks[i]
        traj_collocation.env_reset()
        opt_states_actions = traj_collocation.minimize_collocation()
        (sts, acts) = (opt_states_actions[:H * 3],
                       opt_states_actions[H * 3:])
        opt_states = np.append(opt_states, sts)
        opt_actions = np.append(opt_actions, acts)
        pos = traj_collocation.env.road.vehicles[0]
        traj_collocation.simulate(opt_states_actions)
        traj_collocation.start = (pos.position[0], pos.position[1],
                                  pos.heading)
        i += 1

    opt_states_actions_ls = np.append(opt_states, opt_actions)
    traj_collocation.horizon = H * i
    traj_collocation.start = start_pos
    traj_collocation.env_reset()
    if traj_collocation.images:
        VIDEO_NAME = 'animation.avi'
        video = cv2.VideoWriter(VIDEO_NAME, 0, 1, (600, 300))
        for image in traj_collocation.images:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

    # traj_collocation.simulate(opt_states_actions_ls)
