"""
This module implements the forward shooting type trajectory optimization.
"""
import cv2
import numpy as np
import gymnasium as gym
from scipy.optimize import minimize


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
        self.images = []

    def env_reset(self) -> None:
        """
        Resets the env with respect to start and goal position.
        """
        # Resetting complete environment
        self.env.reset(seed=10)
        # Setting the start position
        self.env.road.vehicles[0].heading = self.start[2]
        self.env.road.vehicles[0].position = np.array([self.start[0],
                                                       self.start[1]])
        # Setting the goal position
        self.env.goal.heading = self.start[2]
        self.env.goal.position = np.array([self.goal[0],
                                           self.goal[1]])

        try:
            # Setting the obstacle
            self.env.road.vehicles[1].heading = np.pi/2
            self.env.road.vehicles[1].position = np.array([2.0, 2.0])
        except IndexError:
            pass

    def forward_shooting(self, actions) -> float:
        """
        Finding the cumulative cost of a given sequences of actions.
        """
        self.env_reset()
        actions = actions.reshape(self.horizon,
                                  self.num_actions)
        total_cost = 0
        for action in actions:
            self.env.step(action)
            cur_pos = self.env.road.vehicles[0]
            cost = ((cur_pos.position[0] - self.goal[0])**2 +
                    (cur_pos.position[1] - self.goal[1])**2 +
                    (cur_pos.heading - self.goal[2])**2)
            total_cost += cost  # Cumulative sum of costs calculation
            # self.env.render()  # To render while iteration

        return total_cost

    def minimize_shooting(self, init_actions=None) -> list:
        """
        Minimizing the cost function considering the kinematic constraints.
        """
        if init_actions is None:  # Initializing the actions randomly
            init_actions = np.zeros(shape=(self.horizon * self.num_actions,))

            # Initializing acceleration (Dubins car model)
            init_actions[::2] = np.random.uniform(low=0, high=1,
                                                  size=(self.horizon,))

            # Initializing steering
            init_actions[1::2] = np.random.uniform(low=-1, high=1,
                                                   size=(self.horizon,))

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
        img = self.env.render()
        self.images += [img]


if __name__ == "__main__":
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

    park_env = gym.make("parking-parked-v0")
    traj_shooting = ForwardShooting(park_env, H, start_pos,
                                    land_marks[0])

    opt_actions = np.array([])
    i = 0
    while i < len(land_marks):
        traj_shooting.goal = land_marks[i]
        traj_shooting.env_reset()
        acts = traj_shooting.minimize_shooting()
        opt_actions = np.append(opt_actions, acts)
        pos = traj_shooting.env.road.vehicles[0]
        traj_shooting.simulate(acts)
        traj_shooting.start = (pos.position[0], pos.position[1],
                               pos.heading)
        i += 1

    traj_shooting.horizon = H * i
    traj_shooting.start = start_pos
    traj_shooting.env_reset()
    if traj_shooting.images:
        VIDEO_NAME = 'animation_shooting.avi'
        video = cv2.VideoWriter(VIDEO_NAME, 0, 1, (600, 300))
        for image in traj_shooting.images:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()
