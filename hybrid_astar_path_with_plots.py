import numpy as np
import car
from utils import *
import matplotlib.pyplot as plt
from hybrid_astar import * 

def hybrid_astar_path_with_plots(veh_loc,start_config,goal_config):
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

    xmin = -35
    ymin = -20
    xmax = 35
    ymax = 20
    width = xmax - xmin
    height = ymax - ymin
    rect = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='k', facecolor='none')
    plt.gca().add_patch(rect)
    plt.grid()
    

    # plot start and end configuration
    ang1 = start_conf[2]
    x1 = start_conf[0]
    y1 = start_conf[1]
    arrow_end_x1 = 3 * np.cos(ang1)
    arrow_end_y1 = 3 * np.sin(ang1)
    plt.arrow(x1,y1,arrow_end_x1,arrow_end_y1,width =0.5, head_width=1, head_length=1,color='blue')

    ang2 = goal_conf[2]
    x2 = goal_conf[0]
    y2 = goal_conf[1]
    arrow_end_x2 = 3 * np.cos(ang2)
    arrow_end_y2 = 3 * np.sin(ang2)
    plt.arrow(x2,y2,arrow_end_x2,arrow_end_y2,width =0.5, head_width=1, head_length=1,color='green')

    # plotting obstacles
    box = obs
    for i in range(len(box)):
        xmin = box[i][0]
        ymin = box[i][1]
        xmax = box[i][2]
        ymax = box[i][3]
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='k', facecolor='r')
        plt.gca().add_patch(rect)

    for i in range(len(open)):
        plt.plot(open[i][0],open[i][1],'.')
        plt.pause(0.01)

    # plot Hybrid Astar path
    for i in range(len(path_astar)-1):
        x_curve, y_curve = ([path_astar[i][0],path_astar[i+1][0]],[path_astar[i][1],path_astar[i+1][1]])
        plt.plot(x_curve,y_curve,'b',linewidth=4)
        plt.pause(0.001)

    # plot Dubins path
    for i in range(len(path_dub)-1):
        x_curve, y_curve = ([path_dub[i][0],path_dub[i+1][0]],[path_dub[i][1],path_dub[i+1][1]])
        plt.plot(x_curve,y_curve,'g',linewidth=4)
        plt.pause(0.1)

    
    plt.show()

    path_inv = []
    for i in range(len(total_path)):
        p1 = total_path[i][0]
        p2 = -total_path[i][1]
        p3 = -total_path[i][2]
        path_inv.append((p1,p2,p3))

    return path_inv
