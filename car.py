import numpy as np
from math import pi

class Car():
    def __init__(self,l=0.3,w=0.2,max_phi=pi/5):
        self.l = l
        self.w = w
        self.max_phi = max_phi
        pass

    def astar_step(cur_conf):
        
        next_confs = [(1,2,3),(2,3,4),(3,4,5)]  # [(x1,y1,th1),(),()]
        pass