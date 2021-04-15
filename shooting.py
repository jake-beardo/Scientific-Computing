import numpy as np
from scipy.optimize  import fsolve
from scipy.integrate import solve_ivp
from solver_functions import *
from main import func
from matplotlib import pyplot as plt

'''
    n = 500
    time_sol = solve_ivp(func,(0,100),[1 , 1],t_eval=np.linspace(0,100,n))
    plt.plot(time_sol.t,time_sol.y.T)
    plt.show()

    # scipy.optimize.fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None)
    # newton(f,Df,x0,epsilon,max_iter)
    sht = shooting(func, sols, **kwargs)
    print('shooot', sht)
'''
# sol = newton(func, derivative, x0, epsilon, max_iter)
# sol = newton(lambda sols, ODE: shooting(ODE, sols, **kwargs),[1, 1, 10],0.01,1000)
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
    This runs a for loop that stores all solutions for steps between the target
    value and the initial value of zero.

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
    t_values, sols = solve_ode(vars,tt, ODE, **kwargs)
    print('inteeeee',sols[-1, :] - vars)
    return sols[-1, :] - vars


# phasecondition: dxdt = 0
def get_phase_conditon(ODE, vars, **kwargs):
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
    return np.array([ODE(0, vars,**kwargs)[0]])


def shooting(tt,sols, ODE, **kwargs):
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
    vars = sols[:-1]
    tt = sols[-1]
    print('sols', sols, vars)
    return np.concatenate((integrate(vars, tt, ODE, **kwargs), get_phase_conditon(ODE, vars, **kwargs)))




def find_nearest(array, value):
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
    idx = (np.abs(array - value)).argmin()
    return idx

# newton(t_vals, sols, ODE, t0, np.full(np.shape(vars), 0.01), 1000)
def newton(t_vals, sols ,Df,t0,epsilon,max_iter,**kwargs):
    #t_vals = t_vals.asarray()
    print('here', t_vals)
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    tn = t_vals[-1]
    for n in range(0,max_iter):
        tn_idx = find_nearest(t_vals, tn)
        ftn = sols[tn_idx][0]
        print('the vars', ftn)
        if np.all(abs(ftn) < epsilon):
            print('Found solution after',n,'iterations.')
            return tn
        Dftn = Df(tn,ftn,**kwargs)
        print('the derivative ', Dftn)
        if np.all(Dftn == 0):
            print('Zero derivative. No solution found.')
            return None
        tn = tn - ftn/Dftn
        print('the t', tn)
    print('Exceeded maximum iterations. No solution found.')
    return None





'''
if __name__ == '__main__':
    shooting_main([0.1, 0.1], 100, 200, func, 0.01, '--runge', a=1,b=0.2,d=0.1)
'''
