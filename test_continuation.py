''' Simple test script for the continuation method '''
import numpy as np
import solver_functions
from shooting_functions import *
from continuation_functions import continuation_natural

def hopf_normal(t, u_vals, beta, sigma):
    u1 = beta*u_vals[0]-u_vals[1]+u_vals[0]*(u_vals[0]**2 + u_vals[1]**2)
    u2 = u_vals[0]+beta*u_vals[1]+u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)
    return np.array([u1,u2])

# ^^ vary beta between 0 and 2

def hopf_mod(t, u_vals, beta):
    u1 = beta*u_vals[0]-u_vals[1]+u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)-u_vals[0]*((u_vals[0]**2 + u_vals[1]**2)**2)
    u2 = u_vals[0]+beta*u_vals[1]+u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)-u_vals[1]*((u_vals[0]**2 + u_vals[1]**2)**2)
    return np.array([u1,u2])
# ^^ vary beta between -1 and 2

    # # where theta is the phase
beta = 0.1
tt = 20



def func(x,c):
    return (x**3) - x + c


# The discretisation case
# us, params = continuation_natural(-1, tt, func , 'c', param_step_size=0.1, param_from=-2,param_to=2, step_size=0.01,n=500, rk_e='--euler')
#
# print(us)

# The shooting case
us, params = continuation_natural(np.array([0.1,0.1]), tt, hopf_normal , 'beta',discretisation=shooting, param_step_size=0.1, param_from=0,param_to=2, step_size=0.01, rk_e='--runge',period_guess=1,sigma=-1,beta=beta)

# continuation_main(init_guess, tt, ODE, init_param, param_step_size=0.1, num_param_guesses=10, step_size=0.01,n=500, rk_e='--runge', **kwargs):

# print(continuation_natural(np.array([0.1,0.1]), tt, hopf_mod , 'beta',discretisation=shooting, param_step_size=0.1, param_from=-1,param_to=2,step_size=0.01,n=500, rk_e='--runge', beta=beta))
# t_vals, sols = solver_functions.solve_ode(us,tt, hopf, beta=beta, sigma=-1)
# plt.plot(t_vals, sols[:,0])
# plt.plot(t_vals, sols[:,1])

# plt.plot(params,us,'r-')
# plt.xlabel("c")
# plt.ylabel('inital conditions')
# plt.show()
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
