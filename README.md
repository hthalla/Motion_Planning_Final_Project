# CPSC 8810: Special Topics: Motion Planning Course Final Project
## Motion Planning for non-Holonomic Agent with Hybrid Astar with Trajectory Optimization

#### File to run:
integrated.py: This file is to be primarily run. The outputs are the text files with the optimized states/actions from Forward Shooting and Direct Collocation algorithms and the simulation videos of the same.

#### Details of Files:

* ##### HybridAStar
    >utils.py: consists of the priority queue and ordered set for A* search.  
    grid.py: discretizes a given region into specified gird cells.  
    car.py: consists of a class car of which our agent is an object, defines its    dimensions and steering angle, consists of a_star step, which calculates the next possible steps of the vehicle.  
    dubins.py: returns the Dubins path for the agent with provided start and end goals.  
    hybrid_astar_path.py: returns the path from the start to the goal node for trajectory optimization.  
    
* #### TrajectoryOptimization
    >forward_shooting.py: this module implements the forward shooting type trajectory optimization.
    direct_collocation.py: this module implements the direct collocation type trajectory optimization.

* #### Outputs
    **Hybrid A star with Forward Shooting**
    ![ForwardShooting](Outputs/animation_shooting.gif)

    **Hybrid A star with Direct Collocation**
    ![DirectCollocation](Outputs/animation_collocation.gif)

#### Dependencies:
* highway_env
* gymnasium
* scipy
* opencv
