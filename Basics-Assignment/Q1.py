import numpy as np
import matplotlib.pyplot as plt
import math

# ode function needed to solve
def ODE(x,t):
    dx_dt = x
    return dx_dt

def sol_x(t):
    solution_x = np.exp(t)
    return solution_x

def euler_step(x_pre, t_pre, ODE, step_size):
    x_new = x_pre + step_size * ODE(x_pre,t_pre)
    t_new = t_pre + step_size
    return x_new, t_new

def do_step(x0,t0,step_size,ODE,n,extra_step):
    x_array = []
    t_array = []
    x_sol = []
    t= t0
    x=x0
    x_n = np.zeros(0,n + 2)
    t_n = np.zeros(0,n + 2)
    x_n[0] = x0
    t_n[0] = t0
    i_count = 0
    for i in range(n):
        x_new, t_new = euler_step(x, t, ODE, step_size)
        x = x_new
        t = t_new
        x_n[i+1] = x
        t_n[i+1] = t
        i_count += 1
    x_extra, t_extra = euler_step(x, t, ODE, extra_step)
    x_n[i_count+1] = x_extra
    t_n[i_count+1] = t_extra

    x_sol.append(x_extra)
    x_array.append(x_n)
    t_array.append(t_n)
    return x_sol,x_array,t_array

def driver(x0,t0,t1,step_size,ODE):
    gap = t1-t0
    n_array = []
    if step_size % gap == 0:
        n = gap/step_size
        extra_step = 0
        x_sol,x_array,t_array = do_step(x0,t0,step_size,ODE,n,extra_step)
    else:
        n = (gap/step_size).astype(int)
        extra_step = gap - n*step_size
        x_sol,x_array,t_array = do_step(x0,t0,step_size,ODE,n,extra_step)
    return x_sol,x_array,t_array


def solve_to(x0,t0,ODE, t2,deltat_max, tol):
    idxs_array = []
    x_array_of_arrays = []
    t_array_of_arrays = []
    x_sol_array = []

    # analytical solution
    x_a = np.arange(0, 1, 0.01)
    y_a = np.zeros(len(x_a))
    for i in range(len(x_a)):
        y_a[i] = sol_x(x_a[i])

    # Creates step_sizes and deletes the ones == 0 or greater than deltalt_max
    n = 21
    step_sizes =  np.linspace(t0,t2,n)
    print("step",step_sizes)
    for j in range(len(step_sizes)):

        if step_sizes[j] > np.float64(deltat_max) or step_sizes[j] == np.float64(0):
            # remove stepsize or dont use stepsize
            idx = np.where(step_sizes==step_sizes[j])
            idxs_array.append(idx)
        else:
            x_sol,xs_array,ts_array = driver(x0,t0,t2,step_sizes[j],ODE)

            x_array_of_arrays.append(xs_array)
            t_array_of_arrays.append(ts_array)
            x_sol_array.append(x_sol)

    step_sizes = np.delete(step_sizes, idxs_array)
    # calculates and store all values of x and t for each different euler step

    return x_array_of_arrays,t_array_of_arrays, step_sizes,x_sol_array

def analytical_sol(t0,t1, step_sizes, sol_x):
    t_a = np.arange(t0, t1, step_sizes)
    x_a = np.zeros(len(t_a))
    for i in range(len(t_a)):
        x_a[i] = sol_x(t_a[i])
    return t_a,x_a

# n is the number of x's wanted between x0 and target solution
def solve_ode(x0,t0,tt, n, ODE,deltat_max, rk_e):
    tol = 1
    t_vals = np.zeros(n)
    x_vals = np.zeros(n)
    x_sols = np.zeros(n)
    t_vals[0] = t0
    x_vals[0] = x0
    x_sols[0] = x0
    steps = int((tt-t0)/n)
    t2 = t0 + steps
    t_vals[1] = t2

    for i in range(len(t_vals)):
        print(i)
        x_arrays,t_arrays, step_sizes,x_sol_array = solve_to(x_vals[i],t_vals[i],ODE, t_vals[i+1],deltat_max,tol)
        print(x_arrays)
        x0 = x_arrays[0][-1][-1]

        t0 = t_arrays[0][-1][-1]
        print("herlllp",t0)
        t2 += steps
        t_vals[i+1] = t0
        x_vals[i+1] = x0
        x_sols[i+1] = x0

    return x_vals, t_vals, x_sols


#sol = odeint(ODE, x0, delta_t, args=(x,t)

# (x0,t0,tt, n, ODE,deltat_max, rk_e) use sys to run
print("hello world", solve_ode(1,0,5, 4, ODE,0.5, "runge"))

#step_sizes =  np.linspace(0,1,101) # stepsize

t_a, x_a = analytical_sol(t0,t2, 0.01, sol_x)
print(max(t_a))
print(t_arrays[2][0])
err = [sol_x(1) - i for i in x_sol_array]
print("err",err)


plt.loglog(step_sizes, err)
plt.ylabel("Error")
plt.xlabel("Step Size")
plt.title("Error in approximation compared to stepsize")
plt.show()
for i in range(len(x_arrays)):
    plt.plot(t_arrays[i][0], x_arrays[i][0], label="approximation")

plt.plot(t_a, x_a, label="analytical")
plt.legend()
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("euler method approximation")
#plt.show()









#def solve_to(ODE,x_pre,deltat_max):
