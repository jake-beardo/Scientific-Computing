''' Simple test script for the shooting method '''
import numpy as np
from shooting import *

def hopf(t, u_vals, beta, sigma):
    u1 = beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]**2 + u_vals[1]**2)
    u2 = u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)
    return np.array([u1,u2])


    # # where theta is the phase
beta = 0.1
tt = 100

us, period = shooting_main(np.array([0.2,-0.1]),tt, hopf, 0.01,1000, 'z', beta=beta, sigma=-1)
theta = period
u_1= np.sqrt(beta)*np.cos(theta)
u_2 = np.sqrt(beta)*np.sin(theta)
tt = 0
if abs(us[0] - np.sqrt(beta)*np.cos(tt+theta)) < 0.1 and abs(us[1] - np.sqrt(beta)*np.sin(tt+theta)) < 0.1:
    print("Test passed")
    print("Your output\nU0: ", us, '\nPeriod: ', period)
    print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
else:
    print("Test failed")
    print("Your output\nU0: ", us, '\nPeriod: ', period)
    print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
