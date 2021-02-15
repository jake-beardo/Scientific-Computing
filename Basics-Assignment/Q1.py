import numpy as np
import matplotlib.pyplot as plt
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

def solve_to(x0,t0,ODE, t2,deltat_max, tol):
    x_array = []
    t_array = []

    # analytical solution
    x_a = np.arange(0, 1, 0.01)
    y_a = np.zeros(len(x_a))
    for i in range(len(x_a)):
        y_a[i] = sol_x(x_a[i])

    x_sol = []
    #
    # NEED TO WORK OUT WHAT TO DO WITH N
    #

    # Creates stepsizes and deletes the ones == 0 or greater than deltalt_max
    n = 11
    step_sizes =  np.linspace(t0,t2,n)
    idxs_array = []
    for j in range(len(step_sizes)):
        print(step_sizes[j],np.float64(t2),np.float64(t2)  %  step_sizes[j])
        if step_sizes[j] > np.float64(deltat_max) or step_sizes[j] == np.float64(0) or np.float64(t2) % step_sizes[j] != 0:
            # remove stepsize
            idx = np.where(step_sizes==step_sizes[j])
            idxs_array.append(idx)
    step_sizes = np.delete(step_sizes, idxs_array)
    print(step_sizes)

    # calculates and store all values of x and t for each different euler step
    for j in range(len(step_sizes)):
        t= t0
        x=x0
        x_n = np.zeros(n + 1)
        t_n = np.zeros(n + 1)
        x_n[0] = x0
        t_n[0] = t0
        for i in range(n):
            x_new, t_new = euler_step(x, t, ODE, step_sizes[j])
            x = x_new
            t = t_new
            x_n[i+1] = x
            t_n[i+1] = t

        x_sol.append(x)
        x_array.append(x_n)
        t_array.append(t_n)

    return x_array,t_array

def analytical_sol(t0,t1, step_sizes, sol_x):
    t_a = np.arange(t0, t1, step_sizes)
    print(t_a)
    x_a = np.zeros(len(t_a))
    for i in range(len(t_a)):
        x_a[i] = sol_x(t_a[i])
    return t_a,x_a




#sol = odeint(ODE, x0, delta_t, args=(x,t)

x0 = 1
t0 = 0
t2 = 1 # Final Value
tol =1
deltat_max = 0.2
x_array,t_array = solve_to(x0,t0,ODE, t2,deltat_max,tol)

#step_sizes =  np.linspace(0,1,101) # stepsize



t_a, x_a = analytical_sol(t0,t2, 0.01, sol_x)
print(max(t_a))
print(max(t_array[2]))

plt.plot(t_a, x_a, label="analytical")
plt.plot(t_array[0], x_array[0], label="analytical")
plt.plot(t_array[2], x_array[2], label="analytical")
plt.show()









#def solve_to(ODE,x_pre,deltat_max):
