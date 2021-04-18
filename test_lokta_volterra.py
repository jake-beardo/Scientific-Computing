''' TEST LOKTA-VOLTERRA SHOOTING AND PLOTTING '''
import numpy as np
from shooting import *


def lokta(t,vars,a,b,d):
    # print('vars are: ',vars)
    return np.array([vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0]), b*vars[1]*(1 - (vars[1]/vars[0]))])



    # # where theta is the phase
tt = 100

# shooting_main(vars,tt, ODE, step_size,n, rk_e, **kwargs)

us, period = shooting_main(np.array([0.1,0.1]),200, lokta, 0.1,500, '--runge', a=1,b=0.2, d=0.1)
print(us,period)
tt = 0
''' NEED TO CHANGE THE CONDITIONS '''
if abs(us[0] - np.sqrt(beta)*np.cos(tt+theta)) < 0.1 and abs(us[1] - np.sqrt(beta)*np.sin(tt+theta)) < 0.1:
    print("Test passed")
    print("Your output\nU0: ", us, '\nPeriod: ', period)
    print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
else:
    print("Test failed")
    print("Your output\nU0: ", us, '\nPeriod: ', period)
    print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
