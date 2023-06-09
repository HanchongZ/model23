# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:42:07 2023

@author: Hanchong Zhu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:54:06 2023

@author: Hanchong Zhu
"""
#------------------------------------------------------------------------------#
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import writers
import random as rand
import ped_utils as putils
#------------------------------------------------------------------------------#
import model_dev as model
#import model
from importlib import reload
reload(model)
#------------------------------------------------------------------------------#
def count_collision(nums):
    count = 0  # Counter for non-zero digit blocks
    current_block = 0  # Counter for the current non-zero digit block
    
    for num in nums:
        if num != 0:
            current_block += 1
        elif current_block != 0:
            count += 1
            current_block = 0
    
    # Check if there is a non-zero digit block at the end of the list
    if current_block != 0:
        count += 1
    
    return count
    
kappa_l = [0, 0.5, 1]
n_l = [12]
occupancy_vals = np.zeros(len(n_l))
avgspd_vals = np.zeros((len(n_l),len(kappa_l)))
collision_vals = np.zeros((len(n_l),len(kappa_l)))
c_l = ["tab:blue","tab:orange","tab:green"]  

for u in range(len(n_l)):
    fig = plt.figure(figsize=(12,6))
    for i in range(len(kappa_l)):
        model.reset_model()
        
        kappa = kappa_l[i]
        
        model.n = n_l[u]
        model.instructions = False # Change True to see errors in initializing parameters before running the model
        model.k = 5000
        model.tau = 0.5
        model.ar = math.radians(0.1)
        model.time_step = 0.1
        
        #Setting half of the pedestrian in the left zone
        np.random.seed(4096)
        model.x = np.zeros((2,model.n))
        model.v = np.zeros((2,model.n))
        destination = np.zeros((2,model.n))
        
        model.x[0,:] = np.linspace(-4,4,model.n+2)[1:-1]
        model.x[1][::] = np.random.uniform( -1.25, 1.25, model.n)
        model.x[0,0] = -4
        model.x[1,0] = 0
        
        model.o = np.zeros((2, model.n))
        model.o[0,:] = 50.
        #150 degree horizon
        model.H_min = math.radians(45)
        model.H_max = math.radians(45)
        model.d_max = 8
        model.mass = np.ones(model.n) * 75 #Person of weight 75
        model.v_0 = np.random.normal( 1.3, 0.2, model.n)
    
        #Initialize the walls [a,b,c,startval,endval]
        model.n_walls = 2
        model.walls = np.empty((5,model.n_walls))
        model.walls[:,0] = np.array([ 0, 1, 1.5, -4, 4])
        model.walls[:,1] = np.array([ 0, 1, -1.5, -4, 4])
        #------------------------------------------------------------------------------#
        model.check_model_ready()
        model.initialize_global_parameters()
        #------------------------------------------------------------------------------#
        collision_mat = np.zeros((model.n,1))
        # stop_time = np.zeros(model.n)
        #Increment the time in steps of relaxation_time
        stoptime = 30
        while model.t < stoptime:
            #print("t={}".format(model.t))
            collision = model.hybrid_model(kappa,return_collision=True)
            collision = np.reshape(collision, (len(collision), 1))
            collision_mat = np.append(collision_mat, collision, axis=1)
            model.update_model()
            # periodic boundary condition
            for j in range(model.n):
                if model.x[0,j] > 4:
                    model.x[0,j] -= 8
                if model.x[0,j] < -4:
                    model.x[0,j] += 8
        
        plt.plot(model.x_full[0, 0, :61],model.x_full[1, 0, :61],"--", color =c_l[i],label="kappa ="+str(kappa))
        for t in [0,12,24,36,48,60]:
            plt.plot(model.x_full[0, 0, t],model.x_full[1, 0, t],"x",color =c_l[i])
        
    plt.legend()
    plt.xlabel("x-axis (m)")
    plt.ylabel("y-axis (m)")
    plt.grid(alpha = 0.5)
    # plt.show()
    plt.savefig('results/test_bi_'+str(n_l[u])+'.png')
    

