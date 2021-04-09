''' Simple test script for the shooting method '''
import numpy as np
from shooting import *

def hopf(t, u_vals, beta, sigma):
    u1 = beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]^2 + u_vals[1]^2)
    return np.array([beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]^2 + u_vals[1]^2),u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]^2 + u_vals[1]^2)])

    # u_vals[0] = sqrt(beta)*np.cos(t+theta)
    # u_vals[1] = sqrt(beta)*np.sin(t+theta)
    # # where theta is the phase

us, period = shooting_main(np.array([1,1]),0,100, hopf, 0.01,'z', beta=1, sigma=-1)
theta = period
if abs(us[0] - sqrt(beta)*np.cos(t+theta)) < 1e-6 and abs(us[1] - sqrt(beta)*np.sin(t+theta)) < 1e-6:
    print("Test passed")
else:
    print("Test failed")
