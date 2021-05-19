import numpy as np
from scipy.optimize  import fsolve
from solver_functions import solve_ode, solve_to, euler_step, rk4
from shooting_functions import shooting, shoot, integrate, get_phase_conditon,period_finder
from pde_solver_functions import pde_solver,forward,backward,crank
from main import func
from matplotlib import pyplot as plt

def append_inits_params(found_inits,init_guess,params_from,param_from):
    found_inits = np.append(found_inits, init_guess)
    params_from = np.append(params_from, param_from)
    return found_inits, params_from

def continuation_natural(init_guess, tt, func, init_param, discretisation=False, param_step_size=0.1, param_from=0,param_to=2, step_size=0.05,n=500, rk_e='--runge',main_param=None,L=0,T=0,mx=0,bound_conds=np.array([0,0]),method=forward,type_bc='Dirichlet',period_guess=None, **kwargs):
    '''
    Varies through a given parameter 'init_param' and finds the inital conditions from param_from to param_to finding all param_step_size's inbetween.

    Depending on the discretisation given the funtion can use natural parameter discretisation or numerical shooting.

    Parameters
    ----------
    init_guess : number/numpy array
        The inital guess for the intial conditons of the function you are doing natual parameter continuation for
    tt  :  number
        The target value of t that the funtion will solve up to
    func :   function
        Differnetial equation or system of differnetial equations. Defined as a function.
    init_param  :   string
        The parameter chosen to do natural parameter continuation for.
    discretisation  :   Boolean/function
        The method you wish to use for natural parameter continuation. Either natural parameter discretisation or numerical shooting.
    param_step_size :   number
        The step size of how much the parameter will change by
    param_from  :   number
        The starting condition of the parameter
    param_to    :   number
        The final value of the parameter you wish to solve for

    Returns
    -------
    found_inits : number/numpy array
        The inital conditons for the function at all of the values of the parameter varied over

    Examples
    --------
    >>> def hopf_mod(t, u_vals, beta):
        u1 = beta*u_vals[0]-u_vals[1]+u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)-u_vals[0]*((u_vals[0]**2 + u_vals[1]**2)**2)
        u2 = u_vals[0]+beta*u_vals[1]+u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)-u_vals[1]*((u_vals[0]**2 + u_vals[1]**2)**2)
        return np.array([u1,u2])
    >>> continuation_natural(np.array([0.1,0.1]), 100, hopf_mod , 'beta',discretisation=shooting, param_step_size=0.1, param_from=-1,param_to=2,step_size=0.01,n=500, rk_e='--runge', beta=0.1)
    [-1. -0.57714843 -0.57735543 ........ -1.50433312 -1.52137971]
    '''
    found_inits = np.array(np.array([]))
    params_from = np.array([])
    kwargs[init_param]=param_from

    #  it simply increments the a parameter by a set amount and attempts to find
    #  the solution for the new parameter value using the last found solution as an initial guess.
    number_of_steps = ((param_to-param_from)/param_step_size)
    if not number_of_steps.is_integer() or number_of_steps < 0:
        raise Exception("number of steps in continuation must be a natural number please choose whole numbers param_from and param_to and/or a different param_step_size")
    for i in range(0,int(number_of_steps)+1):
        # vars,tt, func,step_size=0.01,n=500, rk_e='--runge', **kwargs
        if discretisation==shooting:
            init_guess, period_guess = shooting(init_guess, tt, func, step_size, rk_e,period_guess=period_guess, **kwargs)

            found_inits,params_from = append_inits_params(found_inits,init_guess,params_from,param_from)

        elif discretisation==pde_solver:
            # pde_solver(u_I,L,T,mx,mt,bound_conds,main_param,varied_param=None, method=forward, type_bc='Dirichlet', **kwargs)
            if not main_param:
                raise Exception("You need to pass the main parameter you are using as a string parameter in continuation_natural (e.g. main_param=\'kappa\')")
            u_j,x,t_steady_states = pde_solver(func,L,T,mx,tt,bound_conds,main_param=main_param,varied_param=init_param, method=method, type_bc='Dirichlet',**kwargs)
            if type(t_steady_states[0]) != str:
                init_guess = t_steady_states[0]
                found_inits,params_from = append_inits_params(found_inits,init_guess,params_from,param_from)

        else:
            init_guess = fsolve(func,init_guess,args=kwargs[init_param])
            found_inits,params_from = append_inits_params(found_inits,init_guess,params_from,param_from)

        param_from += param_step_size
        kwargs[init_param]=param_from

    return found_inits, params_from



def pseudo_arc_length_continuation(init_guess, func,n,**kwargs):
    var0,var1 = vars_array

    secant = var1 - var0 # Creating secant
    v_tilda = var1 + secant
    print(v_tilda)
    for i in range(n):
        def rooting_obj_fun(v,**kwargs):
            b = v[0]
            U = v[1:]
            print(v)
            f = func(0,v,**kwargs)
            dot_prod = np.dot((v-v_tilda),secant)
            return np.append(dot_prod,f)

        # fsolve(lambda sols,ODE: shoot(sols, ODE, param_idx,rk_e, **kwargs), inital_guesses,ODE)

        solution = fsolve(lambda v_tilda1: rooting_obj_fun(v_tilda1,**kwargs),v_tilda)
        v_true = solution
        np.append(vars_array,v_true)
        secant = vars_array[-1]-vars_array[-2]
        v_tilda = vars_array[-1] + secant
    final_var = np.array(vars_array)
    return final_var.transpose()
