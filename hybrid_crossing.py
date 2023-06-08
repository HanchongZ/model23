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
n_l = [3,6,10]

for n in n_l:
    fig,ax = plt.subplots(3,3,figsize=(30,30))
    for i in range(len(kappa_l)):
        model.reset_model()
        
        kappa = kappa_l[i]
        
        model.n = n
        model.instructions = False # Change True to see errors in initializing parameters before running the model
        model.k = 1000
        model.tau = 0.5
        model.ar = math.radians(0.1)
        model.time_step = 0.1
        
        #Setting half of the pedestrian in the left zone
        np.random.seed(2023)
        model.x = np.zeros((2,model.n))
        model.o = np.zeros((2,model.n))
        model.v = np.zeros((2,model.n))
        destination = np.zeros((2,model.n))
        for l in range(model.n):
            model.x[0][l] = np.random.uniform(-6, -5)
            model.x[1][l] = np.random.uniform(-1.5, 1.5)
            model.o[0][l] = 8.0
            model.o[1][l] = 1 * model.x[1][l]
            destination[0][l] = 5.0
            destination[1][l] = 1 * model.x[1][l]
            model.v[0][l] = 1.0
        # #Setting half of the pedestrian in the right zone
        # for r in np.arange(model.n//2,model.n):
        #     model.x[0][r] = np.random.uniform(5, 6)
        #     model.x[1][r] = np.random.uniform(-1.5, 1.5)
        #     model.o[0][r] = -10
        #     model.o[1][r] = np.random.uniform(-2.5, 2.5)
        #     model.v[0][r] = -1.0
        #150 degree horizon
        model.H_min = math.radians(75)
        model.H_max = math.radians(75)
        model.mass = np.ones(model.n) * 75 #Person of weight 75
        model.v_0 = 1.3*np.ones(model.n)
    
        #Initialize the walls [a,b,c,startval,endval]
        model.n_walls = 2
        model.walls = np.empty((5,model.n_walls))
        model.walls[:,0] = np.array([ 0, 1, 2.5, -5, 5])
        model.walls[:,1] = np.array([ 0, 1, -2.5, -5, 5])
        #------------------------------------------------------------------------------#
        model.check_model_ready()
        model.initialize_global_parameters()
        #------------------------------------------------------------------------------#
        collision_mat = np.zeros((model.n,1))
        stop_time = np.zeros(model.n)
        #Increment the time in steps of relaxation_time
        while model.t<20:
            #print("t={}".format(model.t))
            collision = model.hybrid_model(kappa,return_collision=True)
            collision = np.reshape(collision, (len(collision), 1))
            collision_mat = np.append(collision_mat, collision, axis=1)
            model.update_model()
            for j in range(model.n):
                distance = np.linalg.norm(model.x[0, j] - destination[0, j])
                if distance < 0.2 and stop_time[j]==0:
                    stop_time[j] = 1 * model.t
            if all(time > 0 for time in stop_time):
                break
        
        avgspdlist = []
        collisionlist = []
        for k in range(model.n):
            spdmat = np.zeros(int(stop_time[k]-0.5/model.time_step))
            spdmat = np.sqrt(model.v_full[0,k,int(0.5/model.time_step):int(stop_time[k]/model.time_step)]**2 + model.v_full[1,k,int(0.5/model.time_step):int(stop_time[k]/model.time_step)]**2)
            avgspdlist.append(np.mean(spdmat))
            
            collisioncount = count_collision(collision_mat[k,int(0.5/model.time_step):int(stop_time[k]/model.time_step)])
            collisionlist.append(collisioncount)
        #------------------------------------------------------------------------------#
        ax[0,i].hist(stop_time,bins=20)
        ax[0,i].set_xlabel("Travel Time")
        ax[0,i].set_ylabel("Frequency")
        ax[0,i].grid(alpha=0.5)
        
        ax[1,i].hist(avgspdlist,bins=20)
        ax[1,i].set_xlabel("Average speed")
        ax[1,i].set_ylabel("Frequency")
        ax[1,i].grid(alpha=0.5)
        
        ax[2,i].hist(collisionlist,bins=10)
        ax[2,i].set_xlabel("Number of collisions")
        ax[2,i].set_ylabel("Frequency")
        ax[2,i].grid(alpha=0.5)
        
        print("total time taken at kappa = {} is {}".format(round(kappa,1),round(model.t,1)))
        
        # ax[i//4,i%4].plot([-12,12],[-3,-3],color='k')
        # ax[i//4,i%4].plot([-12,12],[3,3],color='k')
        # ax[i//4,i%4].legend(loc=1)
        # ax[i//4,i%4].grid(alpha=0.5)
        # ax[i//4,i%4].set_title(r"$\kappa$ = "+ str(i/10))
        # ax[i//4,i%4].set_ylim(-3.6,3.6)
        # ax[i//4,i%4].set_xlim(-12,12)
    
    plt.savefig('results/hybrid_crossing_'+str(n)+'.png')