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
    '''Finds numerical approximation for a function at point t + stepsize ahead.

    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    ODE : function
        Differnetial equation or system of differnetial equations that take vars
        and t values.
    t_pre : number
        Previous or intial t value. Needed in the ODE function to approimate the
        next step.
    step_size : number
        This is the size of the 'step' the euler funtion will approximate using
        e.g it will return the approimation for value(s) of the funtion at
        t_pre + step_size.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    sols : numpy array or number
        The value(s) or approximations of the function at t_pre + step_size. 
    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    sols = vars + step_size * ODE(t_pre,vars,**kwargs)
    return sols

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
    else:
        for i in range(n):
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
        n = int(gap/step_size)
        extra_step = gap - n*step_size
    vars = do_step(vars,t0,step_size,ODE,n,extra_step,rk_e,**kwargs)
    return vars

'''
solve_ode runs a for loop that stores all solutions between the target value
and the initial value.
it then returns an array of independant and dependant variables and solutions
'''
# n is the number of x's wanted between x0 and target solution
def solve_ode(vars,tt, ODE,step_size=0.01,n=500, rk_e='--runge', **kwargs):
    t0 = 0
    sols = [vars]
    t_vals = [t0]
    steps = tt/n
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
