# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:14:33 2023

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
from scipy import stats
import statsmodels.api as sm


def test_significance(plistnear, neardata, plistfar, fardata, publishdata):
       
    # test the slope of near deviance is significant from 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(plistnear, neardata)
    print("p_value for near is "+ str(p_value))

    # test the slope of far deviance is significant from 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(plistfar, fardata)
    print("p_value for far is "+ str(p_value))

    # test the slope of near and far is significantly different
    # Fit a linear regression model for near
    X_line1 = sm.add_constant(plistnear)
    model_line1 = sm.OLS(neardata, X_line1)
    results_line1 = model_line1.fit()

    # Fit a linear regression model for far
    X_line2 = sm.add_constant(plistfar)
    model_line2 = sm.OLS(fardata, X_line2)
    results_line2 = model_line2.fit()

    # Perform a t-test to compare the slopes of the regression lines
    result = sm.stats.ttest_ind(results_line1.resid, results_line2.resid)
    
    print("p_value for near siginificantly different from far is "+ str(result[0]))
    
    def calculate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    inter_pub_near = np.interp(plistnear,[0,3,6],publishdata[0:3])
    inter_pub_far = np.interp(plistfar,[0,3,6],publishdata[5:8])
    
    # Calculate the RMSE
    rmsenear = calculate_rmse(inter_pub_near, neardata)
    rmsefar = calculate_rmse(inter_pub_far, fardata)

    # Print the RMSE
    print("RMSE for near:", rmsenear)
    print("RMSE for far:", rmsefar)
    
    # Calculate the correlation coefficient
    correlation_coefN = np.corrcoef(inter_pub_near, neardata)[0, 1]
    correlation_coefF = np.corrcoef(inter_pub_far, fardata)[0, 1]

    # Print the correlation coefficient
    print("Correlation coefficient near:", correlation_coefN)
    print("Correlation coefficient far:", correlation_coefF)

#------------------------------------------------------------------------------#
plistnear =  np.linspace(0,5,6)#number of perbturbed ped 
ntrial = 50
lateralpositionN = np.zeros((2,ntrial,6))
lateraldevianceN = np.zeros(6)
finalheadingN = np.zeros((ntrial,6))
    
for t in range(6):
    p = int(plistnear[t])
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
        s = rand.sample(range(1,6),p)#index for which virtual ped perturb
        
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

        lateralpositionN[:,nexp,t] = model.x[:,0]
        for q in [-1,-2,-3,-4,-5,-6,-7,-8,-9]:
            finalheadingN[nexp,t] += np.degrees(np.arccos(model.v_full[0,0,q]/np.sqrt(model.v_full[0,0,q]**2 + model.v_full[1,0,q]**2))-np.pi/2)/9
    
meanfhN = np.mean(finalheadingN, axis=0)
avglateralposN = np.mean(lateralpositionN[:,:,0],axis=1)

for t in range(6):
    for nexp in range(ntrial):
        lateraldevianceN[t] += np.linalg.norm(lateralpositionN[:,nexp,t] - avglateralposN)/ntrial

plt.figure(figsize=(8,6))
plt.plot(plistnear,lateraldevianceN,"x--",label="near")

#number of perbturbed ped 
plistfar = np.linspace(0,7,8)
ntrial = 50
lateralpositionF = np.zeros((2,ntrial,8))
lateraldevianceF = np.zeros(8)
finalheadingF = np.zeros((ntrial,8))
    
for t in range(8):
    p = int(plistfar[t])
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
        s = rand.sample(range(6,13),p)#index for which virtual ped perturb
        
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

        lateralpositionF[:,nexp,t] = model.x[:,0]
        for q in [-1,-2,-3,-4,-5,-6,-7,-8,-9]:
            finalheadingF[nexp,t] += np.degrees(np.arccos(model.v_full[0,0,q]/np.sqrt(model.v_full[0,0,q]**2 + model.v_full[1,0,q]**2))-np.pi/2)/9
    
meanfhF = np.mean(finalheadingF, axis=0)
        
avglateralposF = np.mean(lateralpositionF[:,:,0],axis=1)

for t in range(8):
    for nexp in range(ntrial):
        lateraldevianceF[t] += np.linalg.norm(lateralpositionF[:,nexp,t] - avglateralposF)/ntrial
          

plt.plot(plistfar[:-1],lateraldevianceF[:-1],"x--",label="far")

data = np.genfromtxt('datalocalinteract/data_exp1b.csv', delimiter=',')

mean = data[0,1:]
se = data[1,1:]/np.sqrt(10)

test_significance(plistnear, lateraldevianceN, plistfar[:-1], lateraldevianceF[:-1], mean)

plist = [0,3,6,9,12]
plt.plot(plist,mean[:5],color="b",label="Published results near")
plt.fill_between(plist, (mean + 1.96*se)[:5], (mean - 1.96*se)[:5], color="b", alpha=0.15)
plt.plot(plist,mean[5:],color="r",label="Published results far")
plt.fill_between(plist, (mean + 1.96*se)[5:], (mean - 1.96*se)[5:], color="r", alpha=0.15)

plt.legend()
plt.xlabel("Number of perturbed neighbors")
plt.ylabel("Mean lateral deviation (m)")
plt.grid(alpha=0.5)

#-----------------------------------------------------------------------------------

plt.figure(figsize=(8,6))
plt.plot(plistnear,meanfhN,"x--",label="near")
plt.plot(plistfar[:-1],meanfhF[:-1],"x--",label="far")

data = np.genfromtxt('datalocalinteract/data_finalheading.csv', delimiter=',')

mean = data[0,1:]
se = data[1,1:]/np.sqrt(10)

test_significance(plistnear, meanfhN, plistfar[:-1], meanfhF[:-1], mean)

plist = [0,3,6,9,12]
plt.plot(plist,mean[:5],color="b",label="Published results near")
plt.fill_between(plist, (mean + 1.96*se)[:5], (mean - 1.96*se)[:5], color="b", alpha=0.15)
plt.plot(plist,mean[5:],color="r",label="Published results far")
plt.fill_between(plist, (mean + 1.96*se)[5:], (mean - 1.96*se)[5:], color="r", alpha=0.15)

plt.legend()
plt.xlabel("Number of perturbed neighbors")
plt.ylabel("Mean final heading (degrees)")
plt.grid(alpha=0.5)

