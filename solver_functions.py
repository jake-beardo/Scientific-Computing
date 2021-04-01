import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import sys

'''
Function to compute each euler steps
'''
def euler_step(vars, t_pre, ODE, step_size,**kwargs):
    vars = vars + step_size * ODE(t_pre,vars,**kwargs)
    return vars

'''
Finds each RK4 step for the system of ODEs
'''
# N can be outside function make function so its just one step
def rk4(vars, t_pre, ODE, step_size,**kwargs):
    k1 = step_size*ODE(t_pre,vars,**kwargs)
    k2 = step_size*ODE(t_pre+(step_size/2),vars+(k1/2),**kwargs)
    k3 = step_size*ODE(t_pre+(step_size/2),vars+(k2/2),**kwargs)
    k4 = step_size*ODE(t_pre+(step_size),vars+(k3),**kwargs)
    k = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    vars = vars + k
    return vars

'''
do_step does each step and decides whether it needs to do euler or RK4
'''
def do_step(vars,t0,step_size,ODE,n,extra_step,rk_e,**kwargs):
    # returns x_arrays which is the xs between each step
    t = t0
    if rk_e == "--euler":
        for i in range(n):
            vars = euler_step(vars, t, ODE, step_size,**kwargs)

            t += step_size
        vars = euler_step(vars, t, ODE, extra_step,**kwargs)
    elif rk_e == "--runge":
        for i in range(n):

            # rk4(x_pre, t_pre, ODE, step_size)
            vars = rk4(vars, t, ODE, step_size,**kwargs)
            t += step_size
        vars = rk4(vars, t, ODE, extra_step,**kwargs)
    return vars

'''
solve to is the driver function that will ensure each step is filled between
two points. if there is extra space at the end of a step it will compute the
extra step size needed to make the full distance between the two points
'''
def solve_to(vars,t0,ODE, t2,step_size,rk_e,**kwargs):
    gap = t2-t0
    if step_size % gap == 0:
        n = gap/step_size
        extra_step = 0
    else:
        n = (gap/step_size).astype(int)
        extra_step = gap - n*step_size
    vars = do_step(vars,t0,step_size,ODE,n,extra_step,rk_e,**kwargs)
    return vars

'''
solve_ode runs a for loop that stores all solutions between the target value
and the initial value.
it then returns an array of independant and dependant variables and solutions
'''
# n is the number of x's wanted between x0 and target solution
def solve_ode(vars,t0,tt, ODE,step_size, rk_e, **kwargs):
    n = 500
    sols = [vars]
    t_vals = [t0]
    steps = (tt-t0)/n
    for i in range(n+1):
        t2 = t0 + steps
        vars = solve_to(vars, t0, ODE, t2, step_size, rk_e, **kwargs)
        # MIGHT WANT TO CHANGE THIS SO IT JUST ACCEPTS ONE VALUE OF VARS
        sols.append(vars)
        t0 = t2
        t_vals.append(t0)
    sols = np.asarray(sols)
    t_vals = np.asarray(t_vals)
    return t_vals, sols


def main(t0,tt,x0,y0,ODE, deltat_max, step_sizes,**kwargs):
    n = 500
    inits = np.array([x0,y0])
    idxs_array = []
    t_vals_array = []
    sols_array = []
    for j in range(len(step_sizes)):
        if step_sizes[j] > np.float64(deltat_max) or step_sizes[j] == np.float64(0):
            # remove stepsize or dont use stepsize
            idx = np.where(step_sizes==step_sizes[j])
            idxs_array.append(idx)
        else:
            # (x0,t0,tt, n, ODE,deltat_max, rk_e) use sys to run
            t_vals, sols = solve_ode(inits,t0,tt, ODE,step_sizes[j],"--runge",**kwargs)
            t_vals_array.append(t_vals)
            sols_array.append(sols)

    step_sizes = np.delete(step_sizes, idxs_array)

    plt.plot(t_vals_array[0], sols_array[0][:,0], label="runge")
    plt.plot(t_vals_array[0], sols_array[0][:,1], label="runge")
    #plt.plot(max_x, max_y, 'o')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x(t),y(t)")
    plt.show()
    return sols_array[0][:,0], sols_array[0][:,1]
