import numpy as np
import car
from utils import *
import math
import matplotlib.pyplot as plt

car_obj = car.Car()
start_conf = (0,0,0)
goal_conf = (40,20,1)

total_confs = []
# total_confs.append(start_conf)
new_confs = car_obj.astar_step(start_conf)
total_confs.extend(new_confs)
print(new_confs)

for i in range(len(new_confs)):
    n_conf = car_obj.astar_step(new_confs[i])
    total_confs.extend(n_conf)

print(len(total_confs))

for i in range(len(total_confs)):
    ang1 = total_confs[i][2]
    x1 = total_confs[i][0]
    y1 = total_confs[i][1]
    arrow_end_x1 = 0.5 * np.cos(ang1)
    arrow_end_y1 = 0.5 * np.sin(ang1)
    plt.arrow(x1,y1,arrow_end_x1,arrow_end_y1,head_width=0.1, head_length=0.1)
    plt.grid
    # plt.pause(0.1)
plt.show()