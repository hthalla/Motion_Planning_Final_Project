import gymnasium as gym
import numpy as np

env = gym.make("parking-parked-v0")
done = truncated = False
env.configure({
    "screen_height": 600,
    "screen_width": 1200
})

env.reset(seed=10)  # Ensuring we are getting the same scenario

# The below command is only adding a max of 7 vehicles. Need to add manually.
# env.config['vehicles_count'] = 10

# The parked vehicles can be added with the below commands
lane = ("a", "b", 0)
v = env.vehicle.make_on_lane(env.road, lane, 4, speed=0)
env.road.vehicles.append(v)
lane = ("a", "b", 13)
v = env.vehicle.make_on_lane(env.road, lane, 4, speed=0)
env.road.vehicles.append(v)
lane = ("b", "c", 0)
v = env.vehicle.make_on_lane(env.road, lane, 4, speed=0)
env.road.vehicles.append(v)
lane = ("b", "c", 13)
v = env.vehicle.make_on_lane(env.road, lane, 4, speed=0)
env.road.vehicles.append(v)

# Ego vehicle position can be changed as below
env.road.vehicles[0].heading = np.pi/2
env.road.vehicles[0].position = np.array([-30.5, -17.5])

# Parked vehicles can be changed as below
env.road.vehicles[1].heading = np.pi/2
env.road.vehicles[1].position = np.array([2.0, 2.0])

# Setting the goal position
env.goal.heading = np.pi/2
env.goal.position = np.array([14.0, 14.0])

# Controll actions
action = [0.1, 0.1]  # acceleration, steering
obs, reward, done, truncated, info = env.step(action)

# Outputs are as below
# Obs = x, y, vx, vy, cos_h, sin_h
# reward
# done = goal achieved or not
# info = {speed, crashed, action, is_success}

env.render()
