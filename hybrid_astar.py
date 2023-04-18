import numpy as np
import car
import grid
from utils import *


# inputs
# grid,

# output
# path: [(x_s,y_s,th_s),(x1,y1,th1)....(x_g,y_g,th_g)]
def hybrid_astar(grid,start_conf,goal_conf,):
    open_list = []
    closed_list = []
    init_node = start_conf
    cur_node = init_node

    next_confs = car.astar_step(cur_node)   # [(x,y,th)]
    

    pass

def main():
    grid_dimension = [0,0,10,10]
    cell_size = 0.5
    car_obj = car.Car()
    grid_env = grid.Grid(grid_dimension,cell_size) 
    g = grid_env.make_grid()
    print(g)
    print('grid shape:',len(g),len(g[0]))



if __name__== main():
    main()
