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
#------------------------------------------------------------------------------#
# IMPORT DATA
data = np.genfromtxt("data/data_pub_acceleration.csv", delimiter=',')
#------------------------------------------------------------------------------#
#Start plot and plot published data
plt.figure(figsize=(8,6))
plot_pub = plt.plot( data[:,0], data[:,1], 'orangered', linewidth = 2, label = 'Published Results')
plt.axhline(1.3, linewidth = 2, label = 'v_i^0 = 1.3m/s')
#Compute information
timesteps = [0.5, 0.25, 0.1, 0.05]
for timestep in timesteps:
    model.time_step = timestep
    
    model.instructions = False
    model.n = 1
    model.k = 5000
    model.ar = math.radians(0.1)
    model.tau = 0.54
    model.d_max = 10
    model.x = np.zeros((2,model.n))
    model.o = 50*np.ones((2,model.n))
    model.H_min = math.radians(90)
    model.H_max = math.radians(90)
    model.mass = np.ones(model.n) * 75 #Person of weight 75
    model.v_0 = 1.29*np.ones(model.n)
    model.v = np.zeros((2,model.n))
    model.n_walls = 0
    model.walls = np.empty((5,model.n_walls))
    #Initial time, t, set to 0.35 to account for reaction time to start walking
    model.t = 0.35
    model.check_model_ready()
    model.initialize_global_parameters()

    #Increment the time steps
    v_vals = np.array([])
    t_vals = np.array([])
    while (model.t<4):
        v_vals = np.append(v_vals,[np.linalg.norm(model.v[:,0])])
        t_vals = np.append(t_vals,[model.t])
        #compute alpha_des and v_des
        model.compute_destinations()
        model.move_pedestrians()
        model.update_model()

    plt.plot( t_vals, v_vals, linestyle = "--", label = 'time_step = %.2f' % (timestep))
    model.reset_model()
#------------------------------------------------------------------------------#
plt.grid(alpha = 0.5)
plt.legend(loc='best')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.savefig('results/acceleration_timesteps.png')
#------------------------------------------------------------------------------#
