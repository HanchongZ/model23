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
from sklearn.metrics import v_measure_score
#------------------------------------------------------------------------------#
import model_dev as model
#import model
from importlib import reload
reload(model)
#------------------------------------------------------------------------------#
    
kappa_l = [0, 0.5, 1]
v_measure_vals = np.zeros(len(kappa_l))
ntrials = 1

for i in range(len(kappa_l)):
    for n in range(ntrials):
        model.reset_model()
        
        kappa = kappa_l[i]
        
        model.n = 60
        model.instructions = False # Change True to see errors in initializing parameters before running the model
        model.k = 5000
        model.tau = 0.5
        model.ar = math.radians(0.1)
        model.time_step = 0.1
        
        #Setting half of the pedestrian in the left zone
        model.x = np.zeros((2,model.n))
        model.v = np.zeros((2,model.n))
        
        model.x[0,:] = np.random.uniform( -8, 8, model.n)
        model.x[1][::] = np.random.uniform( -2, 2, model.n)
              
        model.o = np.zeros((2, model.n))
        for l in range(model.n//2):
            model.o[0,l] = 50.
        for r in np.arange(model.n//2,model.n):
            model.o[0,r] = -50
        #150 degree horizon
        model.H_min = math.radians(90)
        model.H_max = math.radians(90)
        model.d_max = 10
        model.mass = np.ones(model.n) * 75 #Person of weight 75
        model.v_0 = 1.3 * np.ones(model.n)
    
        #Initialize the walls [a,b,c,startval,endval]
        model.n_walls = 2
        model.walls = np.empty((5,model.n_walls))
        model.walls[:,0] = np.array([ 0, 1, 2, -8, 8])
        model.walls[:,1] = np.array([ 0, 1, -2, -8, 8])
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
            model.hybrid_model(kappa)
            model.update_model()
            # periodic boundary condition
            for j in range(model.n):
                if model.x[0,j] > 8:
                    model.x[0,j] -= 16
                if model.x[0,j] < -8:
                    model.x[0,j] += 16
        
        bounding_area = 4. * 16.
        occupancy_vals = putils.occupancy(model.r,bounding_area)
        print("Occupancy =", occupancy_vals)
        
        #upindex = np.where(model.x[1] > 0)[0]
        #downindex = np.where(model.x[1] <= 0)[0]
        resultlabels = [1 if y > 0 else 0 for y in model.x[1]]
        
        labels = [0] * 30 + [1] * 30
        v_measure_vals[i] += v_measure_score(labels, resultlabels)/ntrials

    print("V-measure:", v_measure_vals[i])
    
    fig,ax = plt.subplots(2,1,figsize=(12,7.5))
    for u in range(model.n//2):
        ax[0].plot(model.x_full[0,u,0],model.x_full[1,u,0],"o",color = 'tab:orange')
        ax[1].plot(model.x_full[0,u,-1],model.x_full[1,u,-1],"o",color = 'tab:orange')
    for d in np.arange(model.n//2,model.n):
        ax[0].plot(model.x_full[0,d,0],model.x_full[1,d,0],"o",color = 'tab:blue')
        ax[1].plot(model.x_full[0,d,-1],model.x_full[1,d,-1],"o",color = 'tab:blue')
    ax[0].plot([-8,8],[-2,-2],color='k')
    ax[0].plot([-8,8],[2,2],color='k')
    ax[1].plot([-8,8],[-2,-2],color='k')
    ax[1].plot([-8,8],[2,2],color='k')
    ax[0].set_title("Time = 0 s")
    ax[1].set_title("Time = "+str(stoptime)+"s, average V-measure ="+ f"{v_measure_vals[i]:3f}")
    # plt.show()
    plt.savefig('results/hybrid_segregation_bi_'+str(kappa)+'.png')
        