''' Simple test script for the shooting method '''
import numpy as np
from shooting_functions import *

def hopf(t, u_vals, beta, sigma):
    u1 = beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]**2 + u_vals[1]**2)
    u2 = u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)
    u3 = -u_vals[2]
    return np.array([u1,u2,u3])


    # # where theta is the phase
beta = 0.1
tt = 100

us, period = shooting_main(np.array([0.1,0.1,0.1]),tt, hopf, 0.01,1000, 'z', beta=beta, sigma=-1)
t_vals, sols = solve_ode(us,tt, hopf, beta=beta, sigma=-1)
plt.plot(t_vals, sols[:,0])
plt.plot(t_vals, sols[:,1])
plt.plot(t_vals, sols[:,2])
plt.xlabel("t")
plt.ylabel("x(t),y(t)")
plt.show()
theta = period
u_1= np.sqrt(beta)*np.cos(theta)
u_2 = np.sqrt(beta)*np.sin(theta)
u_3 = np.exp(-tt)
tt = 0
if abs(us[0] - np.sqrt(beta)*np.cos(tt+theta)) < 0.1 and abs(us[1] - np.sqrt(beta)*np.sin(tt+theta)) < 0.1 and abs(us[2] - np.exp(-tt)) < 0.1:
    print("Test passed")
    print("Your output\nU0: ", us, '\nPeriod: ', period)
    print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
else:
    print("Test failed")
    print("Your output\nU0: ", us, '\nPeriod: ', period)
    print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
