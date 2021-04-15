import numpy as np
from scipy.optimize  import fsolve
from scipy.integrate import solve_ivp
from solver_functions import *
from main import func
from matplotlib import pyplot as plt


def shooting_main(vars,tt, ODE, step_size,n, rk_e, **kwargs):

    t_vals, sols = solve_ode(vars,tt, ODE, **kwargs)
    plt.plot(t_vals, sols[:,0])
    plt.plot(t_vals, sols[:,1])
    plt.xlabel("t")
    plt.ylabel("x(t),y(t)")
    plt.show()
    period_guess = period_finder(t_vals, sols)
    #sol = newton(t_vals, sols, ODE, t0, np.full(np.shape(vars), 0.01), 1000,**kwargs)
    sol = fsolve(lambda sols, ODE: shooting(tt, sols, ODE, **kwargs), [vars[0], vars[1], period_guess], ODE)
    vars = sol[:-1]
    tt = sol[-1]
    print('U0: ', vars)
    print('Period: ',tt)
    return vars, tt


def period_finder(ts, sols):
    '''
    Guesses the period of a set of solutions to a function.

    Uses the differences in the peaks of a funciton (i.e. when dx/dt = 0) to find the period of ossilation.
    The function can also guess the period for a data set.


    Parameters
    ----------
    ts : numpy array
        The values of t that have been approimated for given our intial guess
    sols  :  numpy array
        Solutions to the ode we based on inital guess

    Returns
    -------
    ave_period : number
        The guess for the period of the ode.

    Examples
    --------
    >>> ts = np.array([0.1,0.2...,1])
    >>> sols = np.array([5,0...,5])
    >>> period_finder(ts, sols)
    1
    '''
    i_count = 0
    peaks = []
    for i in sols[1:-1,0]:
        if i > sols[i_count,0] and i > sols[i_count+2,0]:
            peaks.append(ts[i_count+1])

        i_count += 1
    peaks = np.asarray(peaks)
    peaks = peaks[3:]
    print('amount of peaks', len(peaks))
    peak_diffs = np.diff(peaks)
    ave_period = np.mean(peak_diffs)
    print('ave_period', ave_period)
    return ave_period




# specific integrate function to return the difference from vars and the final
# point for vector sols
def integrate(vars, tt, ODE, **kwargs):
    '''
    Uses solve_ode to find the differnce in final solution values at tt for ode and the intial solutions at t0

    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    tt  :  number
        The target value of t that the funtion will integrate for.
    ODE :  function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    sols[-1, :] - vars : numpy array
        Difference between the intial approimation and final approimation.
    Examples
    --------
    >>> def ODE(t,x):
            return np.sin(x**2) - (x**3 - 1)/x
    >>> vars = np.array([0.1,0.1])
    >>> tt = 2
    >>> integrate(vars, tt, ODE)
    [1.21813282 1.21813282]
    '''
    t_values, sols = solve_ode(vars,tt, ODE, **kwargs)
    return sols[-1, :] - vars

def ODE(t,x):
    return np.sin(x**2) - (x**3 - 1)/x
vars = np.array([0.1,0.1])
tt = 2
print('here')
print(integrate(vars, tt, ODE))

def get_phase_conditon(ODE, vars, **kwargs):
    '''
    Finds the phase condition of a function i.e. when f'(x) = 0

    Parameters
    ----------
    ODE :  function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess
        or previous step taken.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    np.array([ODE(0, vars,**kwargs)[0]]) : numpy array
        The phasecondition when the ode is equal to zero.

    Examples
    --------
    >>> def ODE(t,x):
            return np.sin(x)
    >>> vars = np.array([0.1,0.1])
    >>> get_phase_conditon(ODE, vars)
    [0.09983342]
    '''
    return np.array([ODE(0, vars,**kwargs)[0]])
def ODE(t,x):
    return np.sin(x)
vars = np.array([0.1,0.1])
print(get_phase_conditon(ODE, vars))

def shooting(tt,sols, ODE, **kwargs):
    '''
    Uses the get_phase_conditon and integrate function to find the better inital guesses for variables and period od the system of ODEs

    Parameters
    ----------
    tt  :  number
        The target value of t that the funtion will solve up to.
    sols : numpy array or number
        The value(s) or approximations of the function from t = 0 to t = tt
    ODE :  function
        Differnetial equation or system of differnetial equations. Defined as a
        function.
    **kwargs : variables
        This may include any additional variables that may be used in the system
        of ODE's.

    Returns
    -------
    vars_and_period : numpy array
        Inital conditons needed to solve the function and period of the function

    Examples
    --------
    >>> def ODE(t,x):
            return np.sin(x**2) - (x**3 - 1)/x
    >>> vars = np.array([0.1,0.1])
    >>> tt = 2
    >>> shooting(tt,vars, ODE)
    [0.35865757 9.99999983]
    '''
    vars = sols[:-1]
    tt = sols[-1]
    vars_and_period = np.concatenate((integrate(vars, tt, ODE, **kwargs), get_phase_conditon(ODE, vars, **kwargs)))
    return vars_and_period

def ODE(t,x):
    return np.sin(x**2) - (x**3 - 1)/x
vars = np.array([0.1,0.1])
tt = 2
print('shoot')
print(shooting(tt,vars, ODE))







'''
if __name__ == '__main__':
    shooting_main([0.1, 0.1], 100, 200, func, 0.01, '--runge', a=1,b=0.2,d=0.1)
'''
