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

def solve_to(x1,t1,ODE, t2,step_sizes, delta_t_max,tol):
    for step_size in step_sizes:
        if step_size % t2-t1 == 0:
            idx = np.where(step_sizes==step_size)
            np.delete(step_sizes, idx)


#sol = odeint(ODE, x0, delta_t, args=(x,t)

x0 = 1
t0 = 0
tn = 1 # Final Value

#step_sizes =  np.linspace(0,1,101) # stepsize
x_array = []
t_array = []

# analytical solution
x_a = np.arange(0, 1, 0.01)
y_a = np.zeros(len(x_a))
for i in range(len(x_a)):
    y_a[i] = sol_x(x_a[i])

x_sol = []

step_sizes =  np.array([1,0.1,0.01,0.001,0.0001,0.00001])
for j in range(len(step_sizes)):
    x0 = 1
    t0 = 0
    t= t0
    x=x0
    n = ((tn-t0)/step_sizes[j]).astype(int)
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
    print(x)
    x_array.append(x_n)
    t_array.append(t_n)

err = [sol_x(1) - i for i in x_sol]
print("err",err)


plt.loglog(step_sizes, err)
plt.ylabel("Error")
plt.xlabel("Step Size")
plt.title("Error in approximation compared to stepsize")

plt.show()

plt.plot(x_a, y_a, label="analytical")
plt.plot(t_array[0], x_array[0], label="analytical")
plt.plot(t_array[2], x_array[2], label="analytical")



solve_to(0,0,ODE, 1,step_sizes, 1,1)

x_approx = []











#def solve_to(ODE,x_pre,delta_t_max):
