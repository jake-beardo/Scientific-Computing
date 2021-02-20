import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# ode function needed to solve
def x_double_dot(x,t):
    d2x_dt2 = -x
    return d2x_dt2

def x_dot(x,y):
    dx_dt = y
    return dx_dt

def y_dot(x,y):
    dy_dt = -x
    return dy_dt

def sol_x(t):
    solution_x = np.exp(t)
    return solution_x

def euler_step(x_pre, t_pre, ODE, step_size):
    x_new = x_pre + step_size * ODE(x_pre,t_pre)
    t_new = t_pre + step_size
    return x_new, t_new


# N can be outside function make function so its just one step
def rk4(x_pre, t_pre, ODE, step_size):
    k1 = step_size*ODE(x_pre,t_pre)
    k2 = step_size*ODE(x_pre+(k1/2),t_pre+(step_size/2))
    k3 = step_size*ODE(x_pre+(k2/2),t_pre+(step_size/2))
    k4 = step_size*ODE(x_pre+(k3),t_pre+(step_size))
    k = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    x_new = x_pre + k
    return x_new


def do_step(x0,t0,step_size,ODE,n,extra_step,rk_e):
    # returns x_arrays which is the xs between each step
    print(rk_e)
    if rk_e == "--euler":
        t= t0
        x=x0
        x_n = np.zeros(n + 2)
        t_n = np.zeros(n + 2)
        x_n[0] = x0
        t_n[0] = t0
        i_count = 0
        for i in range(n):

            x_new, t_new = euler_step(x, t, ODE[0], step_size)
            x = x_new
            t = t_new
            x_n[i+1] = x
            t_n[i+1] = t
            i_count += 1
        x_extra, t_extra = euler_step(x, t, ODE, extra_step)
        x_n[i_count+1] = x_extra
        t_n[i_count+1] = t_extra

    elif rk_e == "--runge":
        t = t0
        x = x0
        for i in range(n):
            # rk4(x_pre, t_pre, ODE, step_size)
            x = rk4(x, t, ODE[], step_size)
            t += step_size

        x_extra = rk4(x, t, ODE, extra_step)

    x_sol = x_extra
    return x_sol

def driver(x0,t0,t1,step_size,ODE,rk_e):
    # calculates the size of the extra step needed
    gap = t1-t0
    if step_size % gap == 0:
        n = gap/step_size
        extra_step = 0
        x_sol = do_step(x0,t0,step_size,ODE,n,extra_step,rk_e)
    else:
        n = (gap/step_size).astype(int)
        extra_step = gap - n*step_size
        x_sol = do_step(x0,t0,step_size,ODE,n,extra_step,rk_e)
    return x_sol


def solve_to(x0,t0,ODE, t2,deltat_max,step_size, tol,rk_e):
    x_array_of_arrays = []
    t_array_of_arrays = []

    # analytical solution
    x_a = np.arange(t0, t2, 0.01)
    y_a = np.zeros(len(x_a))
    for i in range(len(x_a)):
        y_a[i] = sol_x(x_a[i])

    x_sol = driver(x0,t0,t2,step_size,ODE,rk_e)
    # calculates and store all values of x and t for each different euler step

    return x_array_of_arrays,t_array_of_arrays,x_sol

def analytical_sol(t0,t1, step_sizes, sol_x):
    t_a = np.arange(t0, t1, step_sizes)
    x_a = np.zeros(len(t_a))
    for i in range(len(t_a)):
        x_a[i] = sol_x(t_a[i])
    return t_a,x_a

# n is the number of x's wanted between x0 and target solution
def solve_ode(x0,t0,tt, n, ODE,deltat_max,step_size, rk_e):
    tol = 1
    t_vals = np.zeros(n+3)
    x_sols = np.zeros(n+2)
    t_vals[0] = t0
    x_sols[0] = x0
    steps = int((tt-t0)/n)
    t2 = t0 + steps
    t_vals[1] = t2
    i = 0
    while i <= 4:
        x_arrays,t_arrays,x_sol = solve_to(x_sols[i],t_vals[i],ODE, t_vals[i+1],deltat_max,step_size,tol,rk_e)
        x0 = x_sol
        t2 += steps
        t_vals[i+2] = t2
        x_sols[i+1] = x0
        i += 1

    t_vals = t_vals[:-1]
    return t_vals, x_sols

def error_finder(x_sols_array,t_vals_array,sol_x):
    first_counter = 0
    error_arrays = []
    for sols in x_sols_array:
        err = abs(sol_x(t_vals_array[first_counter][-1]) - sols[-1])
        error_arrays.append(err)
        first_counter += 1
    return error_arrays


#sol = odeint(ODE, x0, delta_t, args=(x,t)

t0,tt = 0,5
x0 = 1
n = 4
deltat_max = 0.5

ns = 20001
step_sizes =  np.linspace(t0,(t0 + int((tt-t0)/n)),num=50, base=10)

ODE = x_dot, y_dot

idxs_array = []
t_vals_array = []
x_sols_array = []
x_sols_array_runge = []
for j in range(len(step_sizes)):
    if step_sizes[j] > np.float64(deltat_max) or step_sizes[j] == np.float64(0):
        # remove stepsize or dont use stepsize
        idx = np.where(step_sizes==step_sizes[j])
        idxs_array.append(idx)
    else:
        # (x0,t0,tt, n, ODE,deltat_max, rk_e) use sys to run

        t_vals_runge, x_sols_runge = solve_ode(x0,t0,tt, n, ODE,deltat_max,step_sizes[j],"--runge")
        t_vals, x_sols = solve_ode(x0,t0,tt, n, ODE,deltat_max,step_sizes[j],"--euler")
        t_vals_array.append(t_vals)
        x_sols_array.append(x_sols)
        x_sols_array_runge.append(x_sols_runge)


step_sizes = np.delete(step_sizes, idxs_array)
#step_sizes =  np.linspace(0,1,101) # stepsize



errs = error_finder(x_sols_array,t_vals_array,sol_x)
errs_runge = error_finder(x_sols_array_runge,t_vals_array,sol_x)
print(errs_runge)
plt.loglog(step_sizes, errs, label="euler")
plt.loglog(step_sizes, errs_runge, label="runge")
plt.ylabel("Error")
plt.xlabel("Step Size")
plt.title("Error in approximation compared to stepsize")
plt.legend()
plt.show()

plt.plot(t_vals_array[1], x_sols_array[1], label="euler")
plt.plot(t_vals_array[1], x_sols_array_runge[1], label="runge")
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
