import numpy as np
import car
from utils import *
import matplotlib.pyplot as plt
from hybrid_astar import * 

def hybrid_astar_path(veh_loc,start_config,goal_config):
    grid_dimension = [-35,-19,35,19]
    cell_size = 0.5
    car_obj = car.Car()

    start_conf = start_config
    start_conf[1] = -start_config[1]
    start_conf[2] = -start_config[2]
    start_conf = (start_conf[0],start_conf[1],start_conf[2])
    print(start_conf)

    goal_conf = goal_config
    goal_conf[1] = -goal_conf[1]
    goal_conf[2] = -goal_conf[2]
    goal_conf = (goal_conf[0],goal_conf[1],goal_conf[2])
    print(goal_conf)

    veh_l = 5+1
    veh_w = 2+1
    obs = []

    for i in range(len(veh_loc)):
        x = veh_loc[i][0]
        y = -veh_loc[i][1]
        xmin = x - veh_w/2
        ymin = y - veh_l/2
        xmax = x + veh_w/2
        ymax = y + veh_l/2

        ob = [xmin,ymin,xmax,ymax]
        obs.append(ob)

    output = hybrid_astar(grid_dimension,cell_size,start_conf,goal_conf,car_obj,obs)

    if len(output)<3:
        path_astar, open = output
        path_astar.insert(0,start_conf)
        total_path = path_astar
        path_dub = []
    else:
        path_astar, path_dub, open = hybrid_astar(grid_dimension,cell_size,start_conf,goal_conf,car_obj,obs)
        path_astar.append(path_dub[0])
        path_astar.insert(0,start_conf)
        total_path = []
        total_path = path_astar + path_dub

    path_inv = []
    for i in range(len(total_path)):
        p1 = total_path[i][0]
        p2 = -total_path[i][1]
        p3 = -total_path[i][2]
        path_inv.append((p1,p2,p3))

    return path_inv
