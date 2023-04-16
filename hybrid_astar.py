import numpy as np
import car


# input
# obs: [(x_min,ymin,x_max,y_max),....]
# grid_dim: [x_min,ymin,x_max,y_max]

# output: grid[[0,0,...1,1,0],[],[]] 2d array of size m*n

def make_grid(obs,grid_dim):
    pass


car = car.Car()


# inputs
# grid,

# output
# path: [(x_s,y_s,th_s),(x1,y1,th1)....(x_g,y_g,th_g)]

def hybrid_astar(grid,start_conf,goal_conf,):
    open_list = []
    closed_list = []
    init_node = start_conf
    cur_node = init_node

    next_confs = car.astar_step(cur_node)
    

    pass