import numpy as np
import car
import grid
from utils import *
import math
import matplotlib.pyplot as plt


# inputs
# grid,

# output
# path: [(x_s,y_s,th_s),(x1,y1,th1)....(x_g,y_g,th_g)]
def hybrid_astar(grid,start_conf,goal_conf,car):
    open_list = PriorityQueue(order=min, f=lambda v: v.f)
    closed_list = OrderedSet()
    init_node = start_conf
    cur_node = init_node

    next_confs = car.astar_step(cur_node)   # [(x,y,th)]
    

    return next_confs

def valid_config(loc, grid_dim):
    conf = []
    x_min = grid_dim[0]
    y_min = grid_dim[1]
    x_max = grid_dim[2]
    y_max = grid_dim[3]
    for pt in loc:
        if pt[0] >= x_min and pt[0] <= x_max and pt[1] >= y_min and pt[1] <= y_max:
            conf.append(pt)            
    return conf

def main():
    grid_dimension = [0,0,10,10]
    cell_size = 0.5
    car_obj = car.Car()
    grid_env = grid.Grid(grid_dimension,cell_size) 
    g = grid_env.make_grid()
    # print(g)
    # print('grid shape:',len(g),len(g[0]))
    start_c = [0,0,0]

    new_confs = hybrid_astar(g,start_c,[1,1,1],car_obj)
    new_confs = valid_config(new_confs, grid_dimension)
    # new_conf_disc = 

    print(new_confs)

    print(math.degrees(new_confs[0][2]))
    print(math.degrees(new_confs[1][2]))
    print(math.degrees(new_confs[2][2]))

    disp = math.sqrt(new_confs[0][0]**2 + new_confs[0][1]**2)
    print(disp)
    plt.scatter(start_c[0],start_c[1])
    plt.scatter(new_confs[0][0],new_confs[0][1])
    plt.scatter(new_confs[1][0],new_confs[1][1])
    plt.scatter(new_confs[2][0],new_confs[2][1])
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.grid()
    plt.show()
if __name__== main():
    main()
    
