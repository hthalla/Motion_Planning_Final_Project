import numpy as np
import car


# input
# obs: [(x_min,ymin,x_max,y_max),....]
# grid_dim: [x_min,ymin,x_max,y_max]
# grid_size: float

# output: grid[[0,0,...1,1,0],[],[]] 2d array of size m*n
def make_grid(grid_dim,cell_size):
    lx = abs(grid_dim[0]-grid_dim[2])
    ly = abs(grid_dim[1]-grid_dim[3])
    nx = int(lx/cell_size)
    ny = int(ly/cell_size)
    grid = [[0 for i in range(nx)] for j in range(ny)]

    return grid


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

    next_confs = car.astar_step(cur_node)   # [(x,y,th)]
    

    pass

def main():
    grid_dimension = [0,0,10,10]
    cell_size = 0.5
    grid = make_grid(grid_dimension,cell_size)
    print(grid)
    print('grid shape:',len(grid),len(grid[0]))



if __name__== main():
    main()
