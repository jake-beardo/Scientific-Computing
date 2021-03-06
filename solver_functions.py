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

def euler_step(vars, t_pre, ODE, step_size):
    x_new = vars[0] + step_size * ODE(vars,t_pre)
    y_new = vars[1] + step_size * ODE(vars,t_pre)
    return [x_new,y_new]


# N can be outside function make function so its just one step
def rk4(vars, t_pre, ODE, step_size):
    k1 = step_size*ODE(vars,t_pre)
    k2 = step_size*ODE(vars+(k1/2),t_pre+(step_size/2))
    k3 = step_size*ODE(vars+(k2/2),t_pre+(step_size/2))
    k4 = step_size*ODE(vars+(k3),t_pre+(step_size))
    k = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    y_new = vars[1] + k
    x_new = vars[0] + k
    return [x_new,y_new]


def do_step(vars,t0,step_size,ODE,n,extra_step,rk_e):
    # returns x_arrays which is the xs between each step
    t = t0
    if rk_e == "--euler":
        for i in range(n):
            y_new = euler_step(vars, t, ODE[1], step_size)[1]
            x_new = euler_step(vars, t, ODE[0], step_size)[0]
            vars[1] = y_new
            vars[0] = x_new
            t += step_size
        y_extra = euler_step(vars, t, ODE[1], extra_step)[1]
        x_extra = euler_step(vars, t, ODE[0], extra_step)[0]
    elif rk_e == "--runge":
        for i in range(n):
            # rk4(x_pre, t_pre, ODE, step_size)
            y = rk4(vars, t, ODE[1], step_size)[1]
            x = rk4(vars, t, ODE[0], step_size)[0]
            vars[1] = y
            vars[0] = x
            t += step_size
        y_extra = rk4(vars, t, ODE[1], extra_step)[1]
        x_extra = rk4(vars, t, ODE[0], extra_step)[0]
    t += extra_step
    vars[1] = y_extra
    vars[0] = x_extra
    return vars


def solve_to(vars,t0,ODE, t2,step_size,rk_e):
    gap = t2-t0
    if step_size % gap == 0:
        n = gap/step_size
        extra_step = 0
        vars = do_step(vars,t0,step_size,ODE,n,extra_step,rk_e)
    else:
        n = (gap/step_size).astype(int)
        extra_step = gap - n*step_size
        vars = do_step(vars,t0,step_size,ODE,n,extra_step,rk_e)

    return vars[0],vars[1]

# n is the number of x's wanted between x0 and target solution
def solve_ode(inits,t0,tt, n, ODE,step_size, rk_e):
    t_vals = np.zeros(n+3)
    x_sols = np.zeros(n+2)
    y_sols = np.zeros(n+2)
    t_vals[0] = t0
    x_sols[0] = inits[0]
    y_sols[0] = inits[1]
    print(tt,t0,n)
    steps = int((tt-t0)/n)
    t2 = t0 + steps
    t_vals[1] = t2
    for i in range(n+1):
        vars = [x_sols[i],y_sols[i]]
        x_sol,y_sol = solve_to(vars,t_vals[i],ODE, t_vals[i+1],step_size,rk_e)
        x0 = x_sol
        y0 = y_sol
        t2 += steps
        t_vals[i+2] = t2
        x_sols[i+1] = x0
        y_sols[i+1] = y0
    t_vals = t_vals[:-1]
    return t_vals, x_sols,y_sols

def error_finder(x_sols_array,t_vals_array,sol_x):
    first_counter = 0
    error_arrays = []
    for sols in x_sols_array:
        err = abs(sol_x(t_vals_array[first_counter][-1]) - sols[-1])
        error_arrays.append(err)
        first_counter += 1
    return error_arrays


#sol = odeint(ODE, x0, delta_t, args=(x,t)

def main(t0,tt,x0,y0,ODE, deltat_max, step_sizes):
    inits = [x0,y0]
    n = tt-1
    print(n)
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

            t_vals_runge, x_sols_runge, y_sols_runge = solve_ode(inits,t0,tt, n, ODE,step_sizes[j],"--runge")
            t_vals, x_sols,y_sols = solve_ode(inits,t0,tt, n, ODE,step_sizes[j],"--euler")
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
    plt.show()
    for i in range(len(x_sols_array)):
        plt.plot(t_vals_array[i], x_sols_array[i], label="approximation")
        plt.plot(t_vals_array[i], x_sols_array_runge[i], label="approximation")

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("euler method approximation")
    errs = error_finder(x_sols_array,t_vals_array,sol_x)
    errs_runge = error_finder(x_sols_array_runge,t_vals_array,sol_x)
    print(errs_runge)
    plt.loglog(step_sizes, errs, label="euler")
    plt.loglog(step_sizes, errs_runge, label="runge")
    plt.ylabel("Error")
    plt.xlabel("Step Size")
    plt.title("Error in approximation compared to stepsize")
    plt.legend()
