import numpy as np
from scipy.optimize  import fsolve
from solver_functions import solve_ode, solve_to, euler_step, rk4
from shooting_functions import shooting_main, shooting, integrate, get_phase_conditon,period_finder
from main import func
from matplotlib import pyplot as plt

def continuation_natural(init_guess, tt, ODE, init_param, discretisation=False, param_step_size=0.1, param_from=0,param_to=2, step_size=0.01,n=500, rk_e='--runge', **kwargs):
    found_inits = np.array(np.array([init_guess]))
    #  it simply increments the a parameter by a set amount and attempts to find
    #  the solution for the new parameter value using the last found solution as an initial guess.
    number_of_steps = ((param_to-param_from)/param_step_size)
    if not number_of_steps.is_integer() or number_of_steps < 0:
        raise Exception("number of steps in continuation must be a natural number please choose whole numbers param_from and param_to and/or a different param_step_size")
    for i in range(0,number_of_steps+1):
        # vars,tt, ODE,step_size=0.01,n=500, rk_e='--runge', **kwargs
        if not discretisation:
            init_guess, period = shooting_main(init_guess, tt, ODE, step_size=0.01,n=500, rk_e='--runge', **kwargs)
        else:
            # lambda x: x**3 - x + c
            # do something different here
            #solve_ode(vars,tt, ODE,step_size=0.01,n=500, rk_e='--runge', **kwargs)
            solve_ode(init_guess,tt, discretisation,step_size,n, rk_e)
        found_inits = np.append(found_inits,np.array([init_guess]),axis=0)
        print('found',found_inits)
        kwargs[init_param] += param_step_size
    return found_inits
