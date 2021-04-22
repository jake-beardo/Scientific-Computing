import numpy as np
from solver_functions import *



#def func(t,vars,a,b,d):
    # print('vars are: ',vars)
#    return np.array([vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0]), b*vars[1]*(1 - (vars[1]/vars[0]))])

def func(t,vars,a,b,d):
    # print('vars are: ',vars)
    return np.array([vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0]), b*vars[1]*(1 - (vars[1]/vars[0]))])

def hopf(t, u_vals, beta, sigma):
    u1 = beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]**2 + u_vals[1]**2)
    u2 = u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]**2 + u_vals[1]**2)
    return np.array([u1,u2])
# for the shooting function we need to have a target eg y0  = 2pi
# then we need to test whehter our predicition lands on that target
# if prediction is over we chose midpoint of predictions.

# set dy/dt(0) = 0 when shooting
# i.e. ODE(0,0,**kwargs) = 0

if __name__ == "__main__":
    #
    # x0s = np.linspace(0.1,0.3,10)
    # y0s = np.linspace(0.1,0.3,10)
    # sols = []
    #
    # for x0 in x0s:
    #     for y0 in y0s:
    # variables required for the funtion
    a = 1
    d = 0.1
    b = 0.2

    tt = 200 # target t value you wish to solve for
    ODE = hopf # ODE you wish to solve for


    inital_conditions, period = shooting_main(vars,tt, ODE, step_size,n, rk_e, **kwargs)
    t_vals, sols = solve_ode(np.array([1,1]),tt, ODE, step_size, 1000, 0.1, beta=1,sigma=-1 )



    print(t_vals, sols)
    plt.plot(t_vals, sols[:,0])
    plt.plot(t_vals, sols[:,1])

    plt.xlabel("t")
    plt.ylabel("x(t),y(t)")
    plt.show()

        # #sols.append(x_sol)
        # #plt.scatter(bs,sols)
        # plt.xlabel('Values of b')
        # plt.ylabel('Prey(x(t_end))')
