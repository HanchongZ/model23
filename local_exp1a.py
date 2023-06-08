# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:40:48 2023

@author: Hanchong Zhu
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random as rand
#------------------------------------------------------------------------------#
import model_dev as model
from importlib import reload
reload(model)
#------------------------------------------------------------------------------#
plist =  [0,3,6,9,12]#number of perbturbed ped 
ntrial = 50
lateralposition = np.zeros((2,ntrial,5))
lateraldeviance = np.zeros(5)
x_full = []

for t in range(5):
    p = plist[t]
    for nexp in range(ntrial):
        model.reset_model()    
        model.instructions = False
        model.n =  13 # one experiment, 5 near, 7 far
        model.time_step = 0.01
        model.tau = 0.5
        model.d_max = 5
        model.H_min = np.pi/2
        model.H_max = np.pi/2
        model.v_0 = 1.3*np.ones(model.n)
        model.k = 1000.0
        model.mass = np.ones(model.n) * 75
        s = rand.sample(range(1,13),p)#index for which virtual ped perturb
        
        #model initial position
        model.x = np.zeros( ( 2, model.n))
        #central position of pedestrain
        model.x[0][0] = 0
        model.x[1][0] = 0
        angler15 = np.linspace(0.25*np.pi,0.75*np.pi,5)
        angler35 = np.linspace(0.25*np.pi,0.75*np.pi,7)
        radiusr15 =  np.ones(5) * 1.5   
        radiusr35 =  np.ones(7) * 3.5    
        for i in range(5):
            angler15[i] = np.random.normal(angler15[i],math.radians(8))
            radiusr15[i] = np.random.normal(radiusr15[i],0.15)
        for j in range(7):
            angler35[j] = np.random.normal(angler35[j],math.radians(8))
            radiusr35[j] = np.random.normal(radiusr35[j],0.15)
        model.x[0][1:6] = np.cos(angler15)*radiusr15 
        model.x[1][1:6] = np.sin(angler15)*radiusr15 
        model.x[0][6:] = np.cos(angler35)*radiusr35 
        model.x[1][6:] = np.sin(angler35)*radiusr35
        
        #all start from rest
        model.v = np.zeros((2,model.n))
        
        #Destination points
        model.o = np.zeros( ( 2, model.n))
        model.o[0][::] = model.x[0][::]
        model.o[1][::] = model.x[1][::] + 12.7
        
        model.n_walls = None
        
        sign = rand.choices((-1,-1),k=p)
        #------------------------------------------------------------------------------#
        model.check_model_ready()
        model.initialize_global_parameters()
        #------------------------------------------------------------------------------#
        while (model.x[1][0] < 12):
            if model.t<=3:
                model.ogive_acceleration(0, 3, 0, 0.5, 1.3)
            else:    
                spdacc,angacc = model.neighbor_interact(0)
                model.neighbor_move_ped(0,spdacc,angacc)
            for i in range(1,13):
                if model.t>=5.5 and i in s:
                    model.set_virtual_destination(i, np.pi/2 - np.radians(10)*sign[s.index(i)], 1.3)
                else:
                    model.set_virtual_destination(i, np.pi/2, 1.3)
                if model.t<=3:
                    model.ogive_acceleration(i, 3, 0, 0.5, 1.3)
                elif model.t>=5 and model.t<5.5 and i in s:
                    model.ogive_perturbation(i, 5, 0.5, 0, 0.083, 0, 10*sign[s.index(i)])
                else:
                    model.move_pedestrians_virtual(i)  
            model.update_model()
            if model.instructions: print(model.t)

        lateralposition[:,nexp,t] = model.x[:,0]
    x_full.append(model.x_full[:,0,:])
avglateralpos = np.mean(lateralposition[:,:,0],axis=1)

for t in range(5):
    for nexp in range(ntrial):
        lateraldeviance[t] += np.linalg.norm(lateralposition[:,nexp,t] - avglateralpos)/ntrial

plt.figure(figsize=(8,6))
plt.plot(plist,lateraldeviance,"x--")
plt.xlabel("Number of perturbed neighbors")
plt.ylabel("Mean lateral deviation (m)")
plt.grid(alpha=0.5)     


