import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# ode function needed to solve

def sol_x(t):
    solution_x = np.exp(t)
    return solution_x

def analytical_sol(t0,t1, step_sizes, sol_x):
    t_a = np.arange(t0, t1, step_sizes)
    x_a = np.zeros(len(t_a))
    for i in range(len(t_a)):
        x_a[i] = sol_x(t_a[i])
    return t_a,x_a

def euler_step(vars, t_pre, ODE, step_size,**kwargs):
    vars = vars + step_size * ODE(vars,t_pre,**kwargs)
    return vars


# N can be outside function make function so its just one step
def rk4(vars, t_pre, ODE, step_size,**kwargs):
    k1 = step_size*ODE(vars,t_pre,**kwargs)
    k2 = step_size*ODE(vars+(k1/2),t_pre+(step_size/2),**kwargs)
    k3 = step_size*ODE(vars+(k2/2),t_pre+(step_size/2),**kwargs)
    k4 = step_size*ODE(vars+(k3),t_pre+(step_size),**kwargs)
    k = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    vars = vars + k
    return vars


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


def solve_to(vars,t0,ODE, t2,step_size,rk_e,**kwargs):
    gap = t2-t0
    if step_size % gap == 0:
        n = gap/step_size
        extra_step = 0
        vars = do_step(vars,t0,step_size,ODE,n,extra_step,rk_e,**kwargs)
    else:
        n = (gap/step_size).astype(int)
        extra_step = gap - n*step_size
        vars = do_step(vars,t0,step_size,ODE,n,extra_step,rk_e,**kwargs)
    return vars

# n is the number of x's wanted between x0 and target solution
def solve_ode(inits,t0,tt, n, ODE,step_size, rk_e, **kwargs):
    x_sols = [inits[0]]
    y_sols = [inits[1]]
    t_vals = [t0]
    vars = inits
    steps = (tt-t0)/n
    print(tt,t0,n,steps)
    t2 = t0 + steps
    for i in range(n+1):
        vars = solve_to(vars, t0, ODE, t2, step_size, rk_e, **kwargs)
        x_sols.append(vars[0])
        y_sols.append(vars[1])
        t0 = t2
        t2 += steps
        t_vals.append(t0)
    return t_vals,x_sols,y_sols

def error_finder(x_sols_array,t_vals_array,sol_x):
    first_counter = 0
    error_arrays = []
    for sols in x_sols_array:
        err = abs(sol_x(t_vals_array[first_counter][-1]) - sols[-1])
        error_arrays.append(err)
        first_counter += 1
    return error_arrays


#sol = odeint(ODE, x0, delta_t, args=(x,t)

def main(t0,tt,x0,y0,ODE,n, deltat_max, step_sizes,**kwargs):
    inits = np.array([x0,y0])
    idxs_array = []
    t_vals_array = []
    x_sols_array = []
    x_sols_array_runge = []
    y_sols_array = []
    y_sols_array_runge = []
    for j in range(len(step_sizes)):
        if step_sizes[j] > np.float64(deltat_max) or step_sizes[j] == np.float64(0):
            # remove stepsize or dont use stepsize
            idx = np.where(step_sizes==step_sizes[j])
            idxs_array.append(idx)
        else:
            # (x0,t0,tt, n, ODE,deltat_max, rk_e) use sys to run
            t_vals_runge, x_sols_runge,y_sols_runge = solve_ode(inits,t0,tt, n, ODE,step_sizes[j],"--runge",**kwargs)
            t_vals, x_sols,y_sols = solve_ode(inits,t0,tt, n, ODE,step_sizes[j],"--euler",**kwargs)
            t_vals_array.append(t_vals)
            x_sols_array.append(x_sols)
            x_sols_array_runge.append(x_sols_runge)
            y_sols_array.append(y_sols)
            y_sols_array_runge.append(y_sols_runge)
    step_sizes = np.delete(step_sizes, idxs_array)
    #step_sizes =  np.linspace(0,1,101) # stepsize

    for i in range(len(t_vals_array)):
        plt.plot(t_vals_array[i], x_sols_array[i], label="euler")
        plt.plot(t_vals_array[i], x_sols_array_runge[i], label="runge")
        plt.plot(t_vals_array[i], y_sols_array[i], label="euler")
        plt.plot(t_vals_array[i], y_sols_array_runge[i], label="runge")


    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("euler method approximation")
    plt.legend()
    #plt.show()
    '''
    for i in range(len(x_sols_array)):
        plt.plot(t_vals_array[i], x_sols_array[i], label="approximation")
        plt.plot(t_vals_array[i], x_sols_array_runge[i], label="approximation")

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("euler method approximation")
    errs = error_finder(x_sols_array,t_vals_array,sol_x)
    errs_runge = error_finder(x_sols_array_runge,t_vals_array,sol_x)
    plt.loglog(step_sizes, errs, label="euler")
    plt.loglog(step_sizes, errs_runge, label="runge")
    plt.ylabel("Error")
    plt.xlabel("Step Size")
    plt.title("Error in approximation compared to stepsize")
    plt.legend()
    '''
