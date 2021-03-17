import numpy as np
from scipy.optimize  import fsolve
from scipy.integrate import solve_ivp
from ode_methods import solve_ode, solve_to
from predator_prey import Lokta_Volterra
from matplotlib import pyplot as plt

def main():
    time_sol = solve_ivp(Lokta_Volterra,(0,100),[1 , 1],t_eval=np.linspace(0,100,500))
    plt.plot(time_sol.t,time_sol.y.T)
    plt.show()
    sol = fsolve(lambda U, f: shoot(f, U), [1, 1, 10], Lokta_Volterra)
    u0 = sol[:-1]
    T = sol[-1]
    print('U0: ', u0)
    print('Period: ',T)
    time_sol = solve_ivp(Lokta_Volterra,(0,T), u0, t_eval=np.linspace(0,T,500))
    plt.plot(time_sol.t,time_sol.y.T)
    plt.xlabel("time")
    plt.ylabel("population change")
    plt.show()


# specific integrate function to return the difference from u0 and the final
# point for vector U
def integrate_ode(method, f, u0, t0, T):
    x_values, t_values = solve_ode(method, f, u0, t0, T, 500, 0.125)
    return x_values[-1, :] - u0


# phase condition: dxdt = 0
def phase(f, u0):
    return np.array([f(0, u0)[0]])


def shoot(f, U):
    u0 = U[:-1]
    T = U[-1]
    return np.concatenate((integrate_ode('runge_kutta', f, u0, 0, T), phase(f, u0)))


if __name__ == '__main__':
    main()
