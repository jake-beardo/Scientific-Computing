import numpy as np
from scipy.optimize  import fsolve
from scipy.integrate import solve_ivp
from solver_functions import *
from Lokta_Volterra import lokta
from matplotlib import pyplot as plt

    '''
    n = 500
    time_sol = solve_ivp(lokta,(0,100),[1 , 1],t_eval=np.linspace(0,100,n))
    plt.plot(time_sol.t,time_sol.y.T)
    plt.show()

    # scipy.optimize.fsolve(func, x0, args=(), fprime=None, full_output=0, col_deriv=0, xtol=1.49012e-08, maxfev=0, band=None, epsfcn=None, factor=100, diag=None)
    # newton(f,Df,x0,epsilon,max_iter)
    sht = shooting(lokta, sols, **kwargs)
    print('shooot', sht)
    '''
    #sol = newton(lokta, derivative, x0, epsilon, max_iter)
    #sol = newton(lambda sols, ODE: shooting(ODE, sols, **kwargs),[1, 1, 10],0.01,1000)


def main(**kwargs):
    sol = fsolve(lambda sols, ODE: shooting(ODE, sols, **kwargs), [1, 1, 10], lokta)
    vars = sol[:-1]
    tt = sol[-1]
    print('U0: ', vars)
    print('Period: ',tt)
    time_sol= solve_ivp(lambda t, vars: lokta(t, vars, **kwargs),(0,tt), vars, t_eval=np.linspace(0,tt,500))
    #time_sol = solve_ivp(lokta,(0,tt), vars, t_eval=np.linspace(0,tt,500))
    plt.plot(time_sol.t,time_sol.y.T)
    plt.xlabel("time")
    plt.ylabel("population change")
    plt.show()




def newton(f,Df,x0,epsilon,max_iter):
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
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None


# specific integrate function to return the difference from vars and the final
# point for vector sols
def integrate(rk_e, ODE, vars, t0, tt, **kwargs):
    t_values, sols = solve_ode(vars,t0,tt, ODE, 0.125, rk_e, **kwargs)
    return sols[-1, :] - vars


# phasecondition: dxdt = 0
def get_phase_conditon(ODE, vars, **kwargs):
    return np.array([ODE(0, vars,**kwargs)[0]])


def shooting(ODE, sols, **kwargs):
    vars = sols[:-1]
    tt = sols[-1]
    return np.concatenate((integrate('--runge', ODE, vars, 0, tt, **kwargs), get_phase_conditon(ODE, vars, **kwargs)))


if __name__ == '__main__':
    main(a=1,b=0.24,d=0.1)
