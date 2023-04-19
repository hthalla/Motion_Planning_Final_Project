import numpy as np
from math import *

class Car():
    def __init__(self,l=1,w=0.2,max_phi=pi/6,v=1):
        self.l = l
        self.w = w
        self.v = v
        self.max_phi = max_phi
        pass

    def astar_step(self,cur_conf):
        x = cur_conf[0]
        y = cur_conf[1]
        th = cur_conf[2]

        next_confs = []
        actions = [-self.max_phi,0,self.max_phi]
        dt = 1

        for i in range(len(actions)):
            xn = x + self.v*cos(actions[i])*dt
            yn = y + self.v*sin(actions[i])*dt
            thn = th + (self.v/self.l)*tan(actions[i])*dt
            next_confs.append([xn,yn,thn])

        return next_confs