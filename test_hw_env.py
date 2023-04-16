import gymnasium as gym
from matplotlib import pyplot as plt
import pprint
import random

env = gym.make("parking-v0")

random.seed(10)
env.configure({
    
    "manual_control": True,
    "show_trajectories": False,
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        },
    "vehicles_count":15,
    "screen_height": 600,
    "screen_width": 1200
})

pprint.pprint(env.config)


env.reset()
done = False
while not done:
    env.render()
    env.step(env.action_space.sample()) 
    