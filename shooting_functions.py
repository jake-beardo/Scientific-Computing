import numpy as np
from scipy.optimize  import fsolve
from scipy.integrate import solve_ivp
from solver_functions import *
from main import func
from matplotlib import pyplot as plt


def shooting_main(vars,tt, ODE, step_size,n, rk_e, **kwargs):
    '''
    Uses numerical shooting and root finder (fsolve) to find the initial conditons and period of a funciton.

    Will first find solutions to the ode based on inital guess then use this to try improve intial guess.
    NB: inital guess for dependant variables need to be quite good.


    Parameters
    ----------
    vars : numpy array or number
        The value(s) or approximations of the function at either the intial guess or previous step taken.
    tt  :  number
        The target value of t that the funtion will solve up to.
    ODE : function
        Differnetial equation or system of differnetial equations. Defined as a function.
    step_size : number
        This is the size of the 'step' the euler funtion will approximate using e.g it will return the approimation for value(s) of the funtion at
        t_pre + step_size.
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
    vars : numpy array
        Inital conditions for the dependant variables of the system of ODEs.
    tt : number
        Period of the system of ODEs.

    Examples
    --------
    >>> def lokta(t,vars,a,b,d):
        return np.array([vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0]), b*vars[1]*(1 - (vars[1]/vars[0]))])
    >>> shooting_main(np.array([0.1,0.1]),200, lokta, 0.1,500, '--runge', a=1,b=0.2, d=0.1)
    [0.10603874 0.18419065] 20.775315952158223
    '''
    try:
        ODE(vars)
    except IndexError:
        raise Exception('The dimensions provided from the intial guess given are smaller than the dimensions required for the ODE. Please try a different inital guess with larger dimensions')
    if np.shape(vars) != np.shape(ODE(vars)):
        raise Exception('The dimensions provided from the intial guess given are larger than the dimensions required for the ODE. Please try a different inital guess with smaller dimensions')
    # using solve_ode to approximate solution for ode using inital guesses
    t_vals, sols = solve_ode(vars,tt, ODE, **kwargs)
    period_guess = period_finder(t_vals, sols) # finding good guess for period of ode
    inital_guesses = np.append(vars,period_guess)
    warnings.filterwarnings('error')
    try:
        # rooting finding using the shooting method
        sol = fsolve(lambda sols, ODE: shooting(tt, sols, ODE, **kwargs), inital_guesses, ODE)
    except RuntimeWarning:
        raise Exception('The root finder is not able to converge. Please try a different intial condition guesses or different guess for the periodic orbit')
    plt.plot(t_vals, sols[:,0])
    plt.plot(t_vals, sols[:,1])
    plt.xlabel("t")
    plt.ylabel("x(t),y(t)")
    plt.show()
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

def hopf(t, u_vals, beta, sigma):
    u1 = beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]**2 + u_vals[1]**2)
    u2 = u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)
    return np.array([u1,u2])
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
    print('integrate')
    print(ODE,'vars',vars)
    print('tt',tt)

    t_values, sols = solve_ode(vars,tt, ODE, **kwargs)
    print('output ',sols[-1, :] - vars)
    return sols[-1, :] - vars
ans = integrate(np.array([0.1,0.1]),6, hopf,beta=0.1, sigma=-1)
print(ans)

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
    print('sols', sols)
    print('tt shooting', tt)
    vars = sols[:-1]
    tt = sols[-1]
    vars_and_period = np.concatenate((integrate(vars, tt, ODE, **kwargs), get_phase_conditon(ODE, vars, **kwargs)))
    return vars_and_period

def func1(t_pre,vars):
    return np.array([t_pre*(vars[0]+ np.sin(t_pre*np.pi)), vars[1]+np.sin(t_pre*np.pi)])
def func2(vars, t_pre,a):
    return np.array(a*(vars + np.sin(t_pre)))
def func3(vars,t_pre):
    return np.array([3*vars[0]+vars[1]-vars[2], vars[0]+2*vars[1]-vars[2], 3*vars[0] +3*vars[1]-vars[2]])
def func4(t,x):
    return np.sin(x)
def hopf(u_vals, t, beta, sigma):
    return np.array([beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]^2 + u_vals[1]^2),u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]^2 + u_vals[1]^2)])
