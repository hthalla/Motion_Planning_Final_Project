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
def hybrid_astar(grid_dim,cell_size,start_conf,goal_conf,car,obs):
    open_list = PriorityQueue(order=min, f=lambda v: v.f)
    closed_list = OrderedSet()
    init_node = start_conf
    cur_node = init_node
    # grid_dim = [xmin,ymin,xmax,ymax]
    grid_env = grid.Grid(grid_dim,cell_size)
    grid_discr = grid_env.make_grid()
    print('discritized grid size:',len(grid_discr),len(grid_discr[0]))
    
    
    start_conf_discr = discr_cor(start_conf,cell_size) 
    # print('discretized start conf:',start_conf_discr)
    goal_conf_discr = discr_cor(goal_conf,cell_size)
    # print('discretized goal conf:',goal_conf_discr)

    reached_goal = False
    open = []
    closed = []

    h = abs(goal_conf[0] - start_conf[0]) + abs(goal_conf[1] - start_conf[1]) #manhattan dist
    g = 0
    f = g+h
    grid_discr[start_conf_discr[0]][start_conf_discr[1]] = (start_conf,f,None)  #(config,f value, parent conf)
    # print(grid_discr)
    # print(grid_discr[start_conf_discr[0]][start_conf_discr[1]])
    
    
    open_list.put(init_node, Value(f=f,g=g))
    # print(open_list._dict)
    open.append(init_node)
    
    k = 0
    while open_list.__len__() > 0:
    # for i in range(30):
        k = k+1
        # print(k)

        node,val = open_list.pop()
        node_discr = discr_cor(node,cell_size)
        closed.append(node[:2])
        # print('popped node:',node)
        # print('discr node:',node_discr)
        # print('discr goal conf:',goal_conf_discr)
        
        if node_discr == goal_conf_discr:
            closed_list.add(node_discr)    # closed list is list of discrete closed nodes 
            reached_goal = True
            print('goal reached')
            break
        closed_list.add(node_discr)

        next_confs = car.astar_step(node)    
        next_confs = valid_config(next_confs, grid_dim)


        # print(next_confs)
        safe_confs = []
        if len(next_confs)>0:
            for i in range(len(next_confs)):
                if aabb_col(next_confs[i],obs):
                    continue
                else:
                    safe_confs.append(next_confs[i])

        # print(safe_confs)
        for i in range(len(safe_confs)):
            safe_conf_disc = discr_cor(safe_confs[i],cell_size)
            sc_d_x = safe_conf_disc[0]
            sc_d_y = safe_conf_disc[1]
            # print('safe conf',safe_confs[i])
            # print('safe conf discr', sc_d_x,sc_d_y)
            # print('discr safe conf',sc_d_x,sc_d_y)
            # if safe_confs[i] not in closed_list:
            if safe_conf_disc not in closed_list._container:

                sc_x = safe_confs[i][0]
                sc_y = safe_confs[i][1]
                sc_g = val.g + 1 # modify 1 with steering action cost
                # sc_h = abs(goal_conf[0]-sc_x) + abs(goal_conf[1]-sc_y)
                sc_h = np.sqrt((goal_conf[0]-sc_x)**2 + (goal_conf[1]-sc_y)**2)
                # sc_h = 0
                sc_f = sc_g + sc_h
                # sc_f = 0
                
                # if open_list.has(safe_confs[i]):
                if sc_d_x < len(grid_discr) and sc_d_y < len(grid_discr[0]): 
                    if grid_discr[sc_d_x][sc_d_y] != 0:
                        if sc_f < grid_discr[sc_d_x][sc_d_y][1]:
                            grid_discr[sc_d_x][sc_d_y] = (safe_confs[i],f,node) #(config,f value, parent conf)
                            
                    else:
                        open_list.put(safe_confs[i], Value(f=sc_f,g=sc_g))
                        grid_discr[sc_d_x][sc_d_y] = (safe_confs[i],f,node) #(config,f value, parent conf)
                        open.append(safe_confs[i])

    
    path = []

    if reached_goal:
        last_node = grid_discr[goal_conf_discr[0]][goal_conf_discr[1]][0]
        while discr_cor(last_node,cell_size) != start_conf_discr:
            last_node_discr = discr_cor(last_node,cell_size)
            parent_node = grid_discr[last_node_discr[0]][last_node_discr[1]][2]
            # print(last_node,parent_node)
            path.insert(0,last_node)
            last_node = parent_node

    # else: 
    #     print('Path not found')
    #     last_node = 
    #     path.insert(grid_discr[])

    return open, path

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

def aabb(conf,l=2,w=1):
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
        o_xmin = obs[j][0]
        o_ymin = obs[j][1]
        o_xmax = obs[j][2]
        o_ymax = obs[j][3]

        if rob_xmin <= o_xmax and rob_xmax>= o_xmin:
            if rob_ymin <= o_ymax and rob_ymax >= o_ymin:
             return True
            
    return False
    

def main():
    grid_dimension = [0,0,60,60]
    cell_size = 1
    car_obj = car.Car()
    start_conf = (5,0,3*np.pi/4)
    goal_conf = (35,25,np.pi/4)
    obs = [[6,0,10,15],[6,22,10,40],[20,2.5,25,5],[20,10,25,15],[30,30,40,40],[18,18,25,35]]
    open, path = hybrid_astar(grid_dimension,cell_size,start_conf,goal_conf,car_obj,obs)
    # print(path) 

    xmin = -1
    ymin = -1
    xmax = 61
    ymax = 61
    width = xmax - xmin
    height = ymax - ymin
    rect = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='k', facecolor='none')
    plt.gca().add_patch(rect)
    plt.grid()

    ang1 = start_conf[2]
    x1 = start_conf[0]
    y1 = start_conf[1]
    arrow_end_x1 = 3 * np.cos(ang1)
    arrow_end_y1 = 3 * np.sin(ang1)
    plt.arrow(x1,y1,arrow_end_x1,arrow_end_y1,width =0.5, head_width=1, head_length=1,color='red')

    ang2 = goal_conf[2]
    x2 = goal_conf[0]
    y2 = goal_conf[1]
    arrow_end_x2 = 3 * np.cos(ang2)
    arrow_end_y2 = 3 * np.sin(ang2)
    plt.arrow(x2,y2,arrow_end_x2,arrow_end_y2,width =0.5, head_width=1, head_length=1,color='green')


    for i in range(len(obs)):
        xmin = obs[i][0]
        ymin = obs[i][1]
        xmax = obs[i][2]
        ymax = obs[i][3]
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='k', facecolor='r')
        plt.gca().add_patch(rect)

    for i in range(len(open)):
        plt.plot(open[i][0],open[i][1],'.')
        plt.pause(0.0001)

    for i in range(len(path)-1):
        x_curve, y_curve = ([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]])
        plt.plot(x_curve,y_curve,'g')
        plt.pause(0.1)

    # for i in range(len(path)):
    #     ang = path[i][2]
    #     x = path[i][0]
    #     y = path[i][1]
    #     arrow_end_x = 3 * np.cos(ang)
    #     arrow_end_y = 3 * np.sin(ang)
    #     plt.arrow(x,y,arrow_end_x,arrow_end_y,width =0.5, head_width=1, head_length=1,color='blue')
    #     plt.pause(0.1)
    plt.show()



if __name__== main():
    main()
    
