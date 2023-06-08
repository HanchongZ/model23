"""
Yadu Raj Bhageria
Imperial College London
Mathematics Department
CID: 00733164
"""
#------------------------------------------------------------------------------#
import math
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------#
import model_dev as model
from importlib import reload
reload(model)
#------------------------------------------------------------------------------#
# IMPORT DATA
data = np.genfromtxt('data/data_pub_corridor_obstacle.csv', delimiter=',')
#------------------------------------------------------------------------------#
fig = plt.figure(figsize=(8,3))
plot_pub = plt.plot( data[:,0], data[:,1], 'r', linewidth = 2, label = 'Published Results')
#------------------------------------------------------------------------------#

reload(model)
model.reset_model()

model.instructions = False
model.n = 2
model.tau = 0.5
model.ar = math.radians(0.1)
model.time_step = 0.1
#Setting person 0 at (-3.5,+0) and person 1 at (0,0)
model.x = np.zeros((2,model.n))
model.x[0][0] = -7./2
model.x[1][0] = 0.01
model.x[0][1] = -0.1 # Added due to the data extrapolated from the published figure
#Setting destination of person 0 as (7.88/2,0)
model.o = np.zeros((2,model.n))
model.o[0][0] = 7.5/2
#150 degree horizon
model.H_min = math.radians(75)
model.H_max = math.radians(75)
model.mass = np.ones(2) * 70 #Person of weight 70
model.mass[1] = 80 #So that r[1] = 0.25
model.v_0 = 1.3*np.ones(model.n)
model.v_0[1] = 0
#Array to hold current information of speeds
model.v = np.zeros((2,model.n))
model.v[0][0] = 1.0
#Initialize the walls [a,b,c,startval,endval]
model.n_walls = 2
# wall y = 1.75/2
model.walls = np.empty((5,model.n))
model.walls[:,0] = np.array([ 0, 1, 1.75/2, -7.88/2, 7.88/2])
# wall y = -1.75/2
model.walls[:,1] = np.array([ 0, 1, -1.75/2, -7.88/2, 7.88/2])
#------------------------------------------------------------------------------#
model.check_model_ready()
model.initialize_global_parameters()
#------------------------------------------------------------------------------#
#Increment the time in steps of relaxation_time
while (model.t<5.5):
#        result_i = model.compute_alpha_des(0,False,False)
#        model.alpha_des[0] = result_i[0]
#        model.f_alpha_des[0] = result_i[1]
#        model.v_des[0] = min(model.v_0[0], model.f_alpha_des[0]/model.tau)
    model.compute_destinations()
    model.move_pedestrians()
    model.update_model()
    if model.instructions: print(model.t)
#------------------------------------------------------------------------------#
plt.plot(model.o[0][0],model.o[1][0],'kx',color="tab:blue",label="Destination point")
circle = plt.Circle((model.x[0][1],model.x[1][1]),model.r[1],color = "orange",label="Obstacle")
fig.gca().add_artist(circle)
plt.plot(model.x_full[0,0,:],model.x_full[1,0,:], linestyle = "--", label = "Pedestrian")

model.reset_model()
#------------------------------------------------------------------------------#
plt.axhline(1.75/2,color = 'k')
plt.axhline(-1.75/2,color ='k')
#plt.yticks([-1.75/2,0,1.75/2])
plt.grid(alpha=0.5)
plt.legend(loc='lower right')
plt.savefig('results/corridor_obstacle_timesteps.png')
