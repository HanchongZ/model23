import numpy as np
import matplotlib.pyplot as plt
import ped_utils as putils

import time
import model_dev as model
#------------------------------------------------------------------------------#
#n_vals = [6,12,18,24,36,48,60]
#n_vals=[100]
vx_ave = []
n_vals = [96]
random = True
display = True
if not random:
    np.random.seed(100)
# n_vals = [60]
t_iteration_vals = [None] * len(n_vals)
t_vals = [None] * len(n_vals)
occupancy_vals = [None] * len(n_vals)
avgspeed_vals = [None] * len(n_vals)
for index in range(len(n_vals)):
    model.reset_model()
    #Set number of pedestrians
    model.n = n_vals[index]

    model.instructions = False
    #Parameters
    model.time_step = 0.1
    model.d_max = 8
    model.H_min = np.pi/4
    model.H_max = np.pi/4
    model.v_0 = np.random.normal( 1.3, 0.2, model.n)
    model.v = np.zeros((2,model.n))
    model.mass = np.ones(model.n) * 75
    # model.v_0 = np.ones(model.n) * 1.3
    model.k = 5000.0

    #Current position uniformly distributed
    model.x = np.zeros( ( 2, model.n))
    model.x[0,:] = np.linspace(-4,4,model.n+2)[1:-1]
    model.x[1][::] = np.random.uniform( -1.25, 1.25, model.n)

    #Destination points, o, set to (50,0) for all
    model.o = np.zeros( ( 2, model.n))
    model.o[0,:] = 50.
    #Initialize the walls [a,b,c,startval,endval]
    model.n_walls = 2
    model.walls = np.zeros( (5, model.n_walls))
    # wall y = -1.5
    model.walls[:,0] = np.array([ 0, 1, 1.5, -4, 4])
    # wall y = 1.5
    model.walls[:,1] = np.array([ 0, 1, -1.5, -4, 4])

    model.check_model_ready()
    model.initialize_global_parameters()
    #------------------------------------------------------------------------------#
    #Increment the time
    start_time = time.time()
    i=0
    while (model.t<60):
        if model.t%1==0: print("t=%f" %(model.t))
        #print ("t = %.2f" %(model.t))
        #compute alpha_des and v_des for each i
        i+=1
        model.compute_destinations()

        model.move_pedestrians()
        for i in range(model.n):
                if model.x[0,i] > 4:
                    model.x[0,i] = model.x[0,i] - 8
                if model.x[0,i] < -4:
                    model.x[0,i] = model.x[0,i] + 8
        #Update alpha_0 and alpha_current
        model.update_model()
        if model.t == model.time_step:
            t_iteration_vals[index] = time.time() - start_time
            print( "Time for 1 iteration: %.3f" %(t_iteration_vals[index]) )
        if (i-1 % 100) ==0: print("i,t=",i,model.t)
    end_time = time.time()
    vx_ave.append(model.v_full[0,:,:].mean())
    t_vals[index] = end_time - start_time
    print( "Time Taken: %.3f" %(t_vals[index]) )
    #------------------------------------------------------------------------------#
    bounding_area = 3. * 8.
    occupancy_vals[index] = putils.occupancy(model.r,bounding_area)
    print("Occupancy =", occupancy_vals[index])
    avgspeed_vals[index] = putils.average_speed(model.v_full[:,:,20:])
    print("Average Speed = ", avgspeed_vals[index])
    print("Comfortable Walking Speeds =", model.v_0)

    model.reset_model()
#------------------------------------------------------------------------------#
print("\nn:\n", n_vals)
print("Occupancy:\n", occupancy_vals)
print("Average Speed:\n", avgspeed_vals)
print("Time Taken:\n", t_vals)
print("Iteration Times:\n", t_iteration_vals)
#------------------------------------------------------------------------------#
np.savetxt('results/results_occupancy.out', (n_vals, occupancy_vals, avgspeed_vals, t_vals, t_iteration_vals), delimiter = ',')
#------------------------------------------------------------------------------#
# IMPORT DATA
data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
#------------------------------------------------------------------------------#
if display:
    fig = plt.figure()
    plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plot_results = plt.plot(occupancy_vals,avgspeed_vals,'x-', label = 'My Results')
    plt.legend(loc=1)
    plt.title('Occupancy Comparison')
    plt.xlabel('Occupancy')
    plt.ylabel('Avg Speed (m/s)')
    # plt.show()
    plt.savefig('results/c_occupancy.png')
