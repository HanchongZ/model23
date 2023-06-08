# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:35:24 2022

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
from importlib import reload
reload(model)
#------------------------------------------------------------------------------#
p = 6 #number of perbturbed ped

model.reset_model()    
model.instructions = False
model.n =  13
model.time_step = 0.25
model.tau = 0.5
model.d_max = 5
model.H_min = np.pi/2
model.H_max = np.pi/2
model.v_0 = 1.3*np.ones(model.n)
model.k = 5000.0
model.mass = np.ones(model.n) * 75
s = rand.sample(range(1,13),p)

#model initial position
model.x = np.zeros( ( 2, model.n))
#central position of pedestrain
model.x[0][0] = 0
model.x[1][0] = 0
angler15 = np.linspace(0,np.pi,5)
angler35 = np.linspace(0,np.pi,7)
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
    temp = model.x[:][0]
    model.compute_single_destination(0)
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
            model.move_pedestrains_virtual(i)       
    model.move_pedestrains_real(0)    
    model.update_model()
    if model.instructions: print(model.t)

# _,_,l =model.v_full.shape
# t = np.linspace(0,model.t,l)
# plt.figure()
# plt.plot(t,np.sqrt(model.v_full[0,0,:]**2 + model.v_full[1,0,:]**2), linestyle = "--")
# plt.show()
    
_,_,num_steps = model.x_full.shape  
     
x_full = [model.x_full[:,i,:] for i in range(model.n)]

fig, ax = plt.subplots()

lines = [ax.plot([], [])[0] for _ in x_full]


ax.set(xlim=(-5, 5), xlabel='X')
ax.set(ylim=(-1, 13), ylabel='Y')

def update_lines(num, x_full, lines):
    for line, x in zip(lines, x_full):
        line.set_data(x[:, :num])
    return lines

# Creating the Animation object
anim = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(x_full, lines), interval=100)

Writer = writers['ffmpeg']
writer = Writer(fps=15, metadata ={'artisit':'Me'},bitrate=1800)

anim.save('animatedexp.mp4',writer)



