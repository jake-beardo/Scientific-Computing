import numpy as np
from scipy.optimize  import fsolve
from solver_functions import solve_ode, solve_to, euler_step, rk4
from shooting_functions import shooting_main, shooting, integrate, get_phase_conditon,period_finder
from main import func
from matplotlib import pyplot as plt

def continuation_main(init_guess, tt, ODE, init_param, param_step_size=0.1, num_param_guesses=10, step_size=0.01,n=500, rk_e='--runge', **kwargs):
    found_inits = np.array(np.array([init_guess]))
    #  it simply increments the a parameter by a set amount and attempts to find
    #  the solution for the new parameter value using the last found solution as an initial guess.
    for i in range(0,n+1):
        # vars,tt, ODE,step_size=0.01,n=500, rk_e='--runge', **kwargs
        init_guess, period = shooting_main(init_guess, tt, ODE, step_size=0.01,n=500, rk_e='--runge', **kwargs)
        found_inits = np.append(found_inits,np.array([init_guess]),axis=0)
        print('found',found_inits)
        kwargs[init_param] += param_step_size
    print(found_inits)
