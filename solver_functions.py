import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import math
import sys


def euler_step(vars, t_pre, ODE, step_size,**kwargs):
    '''Finds numerical approximation for a function at point t + stepsize ahead.

    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    ODE : function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
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
    >>> def ODE(t,x):
            return x**2
    >>> euler_step(1, 0, ODE, 0.1)
    1.1
    '''
    sols = vars + step_size * ODE(t_pre,vars,**kwargs)
    return sols

'''
Finds each RK4 step for the system of ODEs
'''
# N can be outside function make function so its just one step
def rk4(vars, t_pre, ODE, step_size,**kwargs):
    '''Finds numerical approximation for a function at point t + stepsize ahead.

    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    ODE : function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
    t_pre : number
        Previous or intial t value. Needed in the ODE function to approimate the
        next step.
    step_size : number
        This is the size of the 'step' the euler funtion will approximate using.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    sols : numpy array or number
        The value(s) or approximations of the function at t_pre + step_size.

    Examples
    --------
    >>> def ODE(t,x):
            return np.sin(x)
    >>> rk4(1, 0, ODE, 0.1)
    1.084147098
    '''
    k1 = step_size*ODE(t_pre,vars,**kwargs)
    k2 = step_size*ODE(t_pre+(step_size/2),vars+(k1/2),**kwargs)
    k3 = step_size*ODE(t_pre+(step_size/2),vars+(k2/2),**kwargs)
    k4 = step_size*ODE(t_pre+(step_size),vars+(k3),**kwargs)
    k = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    vars = vars + k
    return vars



def solve_to(vars,t0,ODE, t2,step_size,rk_e,**kwargs):
    '''
    This is a driver function that will ensure each step is filled between
    two points. It then sends the information needed to either
    the euler_step or rk4 funtion.

    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    t0  :  number
        The current/intial value of t that has been approimated for.

    ODE :  function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
    t2 : number
        The value of t you wish to approximate up to.
    step_size : number
        This is the size of the 'step' the euler funtion will approximate using.
    rk_e : string
        String '--euler' chooses to compute step using euler method. Otherwise
        will use 4th order Runge-Kutta method.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    sols : numpy array or number
        The value(s) or approximations of the function at t_pre + step_size.

    Examples
    --------
    >>> def ODE(t,x):
            return np.sin(x)
    >>> solve_to(1,0,ODE, 1, 0.1,'h')
    1.9562947385102594
    '''
    gap = t2-t0
    if gap == 0:
        return vars
    elif step_size % gap == 0:
        n = int(gap/step_size)
        extra_step = 0
    else:
        n = int(gap/step_size)
        extra_step = gap - n*step_size
    t = t0
    if rk_e == "--euler":
        for i in range(n):
            vars = euler_step(vars, t, ODE, step_size,**kwargs)
            t += step_size
        sols = euler_step(vars, t, ODE, extra_step,**kwargs)
    else:
        for i in range(n):
            vars = rk4(vars, t, ODE, step_size,**kwargs)
            t += step_size
        sols = rk4(vars, t, ODE, extra_step,**kwargs)
    return sols
def func2(vars, t_pre,a):
    return np.array(a*(vars + np.sin(t_pre)))
def func1(t_pre,vars):
    return np.array([t_pre*(vars[0]+ np.sin(t_pre*np.pi)), vars[1]+np.sin(t_pre*np.pi)])
print(solve_to(np.array([-10,-11]), 10, func1, 12,0.001,'--runge'))

'''

'''
# n is the number of x's wanted between x0 and target solution
def solve_ode(vars,tt, ODE,step_size=0.01,n=500, rk_e='--runge', **kwargs):
    '''
    This runs a for loop that stores all solutions for steps between the target
    value and the initial value of zero.

    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    tt  :  number
        The target value of t that the funtion will solve up to.
    ODE :  function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
    step_size : number
        This is the size of the 'step' the euler funtion will approximate using.
    n : number
        The number of steps you want to take between the inital value and target
        value. The more steps (i.e. higher n) the better the approimation will
        be. n is set to 500 by defult.
    rk_e : string
        String '--euler' chooses to compute step using euler method. Otherwise
        will use 4th order Runge-Kutta method.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    t_vals : numpy array
        This is the array of t values that have been approimated for.
    sols : numpy array
        The array of values that have been approimated for each corresponding
        t value.

    Examples
    --------
    >>> def ODE(t,x):
            return np.sin(x)
    >>> solve_ode(1,1,ODE)
    (array([0.   , 0.002,......, 1.]), array([1., 1.00168385,
        ,.......,1.94328858]))

    '''
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
