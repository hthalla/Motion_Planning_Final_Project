"""
Integrating hybrid A* with trajectory optimization
"""
import cv2
import numpy as np
import gymnasium as gym
from hybrid_astar_path import hybrid_astar_path
from TrajectoryOptimization.direct_collocation import DirectCollocation

env = gym.make("parking-parked-v0")
env.configure({
    "screen_height": 600,
    "screen_width": 1200
})
env.reset(seed = 10)  # Ensuring we are getting the same scenario

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
env.road.vehicles[0].heading = np.pi/2.1
env.road.vehicles[0].position = np.array([-30.5, -17.5])
ego_pos = env.road.vehicles[0].position
ego_head = env.road.vehicles[0].heading
start_conf = [ego_pos[0], ego_pos[1], ego_head]

# Parked vehicles can be changed as below
env.road.vehicles[1].heading = np.pi/2
env.road.vehicles[1].position = np.array([2.0, 2.0])

# Setting the goal position
env.goal.heading = np.pi/2
env.goal.position = np.array([14.0, 14.0])
goal_conf = [env.goal.position[1],
             env.goal.position[1],
             env.goal.heading]

veh_loc = env.road.vehicles
veh_pos = []
for i, veh in enumerate(veh_loc):
    if i > 0:
        veh_pos.append(veh.position)
path = hybrid_astar_path(veh_pos, start_conf, goal_conf)
print(path)

# %%
H = 3
land_marks = path[1:]
start_pos = np.array([-30.5, -17.5, np.pi/2.1])
traj_collocation = DirectCollocation(env, H, start_pos,
                                     land_marks[0])

opt_states = np.array([])
opt_actions = np.array([])
i = 0
while i < len(land_marks):
    traj_collocation.goal = land_marks[i]
    traj_collocation.env_reset()
    traj_collocation.env.render()
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
traj_collocation.start = start_conf
traj_collocation.env_reset()
if traj_collocation.images:
    VIDEO_NAME = 'animation.avi'
    video = cv2.VideoWriter(VIDEO_NAME, 0, 1, (1200, 600))
    for image in traj_collocation.images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()

print(opt_states_actions_ls)
