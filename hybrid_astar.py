import numpy as np
import car
import grid
from utils import *
import math
import matplotlib.pyplot as plt


def discr_cor(safe_confs, cell_size=0.5):
    sf_x = safe_confs[0]
    sf_y = safe_confs[1]

    ds_x = math.ceil(sf_x / cell_size) - 1
    ds_y = math.ceil(sf_y / cell_size) - 1
    
    if sf_x % cell_size == 0:
        ds_x += 1
    if sf_y % cell_size == 0:
        ds_y += 1
        
    return (ds_x, ds_y)


# inputs
# grid,

# output
# path: [(x_s,y_s,th_s),(x1,y1,th1)....(x_g,y_g,th_g)]
def hybrid_astar(grid_dim,cell_size,start_conf,goal_conf,car):
    open_list = PriorityQueue(order=min, f=lambda v: v.f)
    closed_list = OrderedSet()
    init_node = start_conf
    cur_node = init_node
    # grid_dim = [xmin,ymin,xmax,ymax]
    grid_env = grid.Grid(grid_dim,cell_size)
    grid_discr = grid_env.make_grid()
    
    goal_conf_discr = discr_cor(goal_conf)


    obs = []

    h = abs(goal_conf[0] - start_conf[0]) + abs(goal_conf[1] - start_conf[1]) #manhattan dist
    g = 0
    f = g+h

    open_list.put(init_node, Value(f=f,g=g))
    while len(open_list) > 0:
        node,val = open_list.pop()
        node_discr = discr_cor(node)

        if node_discr == goal_conf_discr:
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

        for i in range(len(safe_confs)):
            if safe_confs[i] not in closed_list:

                sc_x = safe_confs[i][0]
                sc_y = safe_confs[i][1]
                sc_g = val.g + 1 # modify 1 with steering action cost
                sc_h = abs(goal_conf[0]-sc_x) + abs(goal_conf[1]-sc_y)
                sc_f = sc_g + sc_h
                
                if open_list.has(safe_confs[i]):
                    if sc_f < open_list._dict[safe_confs[i]].f:
                        open_list._dict[safe_confs[i]].f = sc_f
                        open_list._dict[safe_confs[i]].g = sc_g
                        # add parent nodes in grid here

                    else:
                        open_list.put(safe_confs[i], Value(f=sc_f,g=sc_g))



    
                    





    next_confs = car.astar_step(cur_node)   # [(x,y,th)]
        
    return next_confs

def valid_config(loc, grid_dim): #checks if a configuration lies outside the grid
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
    parent_g = grid_env.make_grid()
    print(parent_g)
    # path = hybrid_astar(parent_g) 

if __name__== main():
    main()
    
