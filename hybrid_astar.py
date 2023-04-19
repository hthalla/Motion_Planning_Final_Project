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
    grid_dim = [0,0,10,10]
    obs = []

    h = 0
    g = 0
    f = g+h

    open_list.put(init_node, Value(f=f,g=g))
    while len(open_list) > 0:
        node,val = open_list.pop()
        if node == goal_conf:
            closed_list.add(node)
            break
        closed_list.add(node)

        next_confs = car.astar_step(node)    
        next_confs = valid_config(next_confs, grid_dim)
        safe_confs = []
        for i in range(len(next_confs)):
            if aabb_col(next_confs[i],obs):
                continue
        else:
            safe_confs.append(next_confs[i])
        


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

def aabb(conf,l,w):
    x = conf[0]
    y = conf[1]
    th = conf[2]
    rlx = x - (w/2)*math.sin(th)
    rly = y + (w/2)*math.cos(th)
    rrx = x + (w/2)*math.sin(th)
    rry = y - (w/2)*math.cos(th)
    frx = x + l*math.cos(th) + (w/2)*math.sin(th)
    fry = y + l*math.sin(th) - (w/2)*math.cos(th)
    flx = x + l*math.cos(th) - (w/2)*math.sin(th)
    fly = y + l*math.sin(th) + (w/2)*math.cos(th)

    A = [rlx,rly]
    B = [rrx,rry]
    C = [frx,fry]
    D = [flx,fly]

    xmin = min(rlx,rrx,frx,flx)
    ymin = min(rly,rry,fry,fly)
    xmax = max(rlx,rrx,frx,flx)
    ymax = max(rly,rry,fry,fly)

    return xmin,ymin,xmax,ymax



def aabb_col(conf,obs):     # obs = [[xmin,ymin,xmax,ymax],...]
    
    rob_xmin,rob_ymin,rob_xmax,rob_ymax = aabb(conf)
    for j in range(len(obs)):
        o_xmin = obs[0]
        o_ymin = obs[1]
        o_xmax = obs[2]
        o_ymax = obs[3]

        if rob_xmin <= o_xmax and rob_xmax>= o_xmin:
            if rob_ymin <= o_ymax and rob_ymax >= o_ymin:
             return True
            
    return False
    

def main():
    grid_dimension = [0,0,10,10]
    cell_size = 0.5
    car_obj = car.Car()
    grid_env = grid.Grid(grid_dimension,cell_size) 
    g = grid_env.make_grid()
    # print(g)
    # print('grid shape:',len(g),len(g[0]))
    start_c = [0,0,0]
    h = 0
    g = 0
    f = g+h


    new_confs = hybrid_astar(g,start_c,[1,1,1],car_obj)
    new_confs = valid_config(new_confs, grid_dimension)
    
    obs = [] ## define onbstacles here
    safe_confs = []
    for i in range(len(new_confs)):
        if aabb_col(new_confs[i],obs):
            continue
        else:
            safe_confs.append(new_confs[i])



    print(new_confs)

    print(math.degrees(new_confs[0][2]))
    print(math.degrees(new_confs[1][2]))
    # print(math.degrees(new_confs[2][2]))

    disp = math.sqrt(new_confs[0][0]**2 + new_confs[0][1]**2)
    print(disp)
    # plt.scatter(start_c[0],start_c[1])
    # plt.scatter(new_confs[0][0],new_confs[0][1])
    # plt.scatter(new_confs[1][0],new_confs[1][1])
    # # plt.scatter(new_confs[2][0],new_confs[2][1])
    # plt.xlim([-10,10])
    # plt.ylim([-10,10])
    # plt.grid()
    # plt.show()

if __name__== main():
    main()
    
