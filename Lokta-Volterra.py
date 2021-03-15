import numpy as np
from solver_functions import *

def lokta(vars,t,a,b,d):
    return np.array([vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0]), b*vars[1]*(1 - (vars[1]/vars[0]))])

# for the shooting function we need to have a target eg y0  = 2pi
# then we need to test whehter our predicition lands on that target
# if prediction is over we chose midpoint of predictions.

# set dy/dt(0) = 0 when shooting
# i.e. ODE(0,0,**kwargs) = 0

if __name__ == "__main__":

    x0s = np.linspace(0.1,0.3,10)
    y0s = np.linspace(0.1,0.3,10)
<<<<<<< Updated upstream
    bs = np.linspace(0.1,0.5,61)
=======
    bs = [0.1,0.26,0.5]
>>>>>>> Stashed changes
    sols = []
    '''
    for x0 in x0s:
        for y0 in y0s:
    '''
<<<<<<< Updated upstream
    #for b in bs:
    a = 1
    d = 0.1

    #b = np.linspace(0.1,0.5,100)
    #b = 0.1
    t0 = 0
    tt = 200
    n = 500
    step_sizes =  np.logspace(-3,(t0 + int((tt-t0)/n)),2)
    step_sizes = np.array([0.1])
    #step_sizes = np.linspace(0,(t0 + int((tt-t0)/n),5))
    #step_sizes = np.array([0.1,1])
    ODE = lokta
    # main(t0,tt,x0,y0,ODE, deltat_max, step_sizes)
    x_sol = main(t0,tt,0.1,0.1,ODE,n, 0.5, step_sizes,a=1,b=0.2,d=0.1)
    #sols.append(x_sol)
    #plt.scatter(bs,sols)
    plt.xlabel('Values of b')
    plt.ylabel('Prey(x(t_end))')
    #plt.show()
=======
    for b in bs:
        a = 1
        d = 0.1

        #b = np.linspace(0.1,0.5,100)
        #b = 0.1
        t0 = 0
        tt = 500
        n = 1000
        step_sizes =  np.logspace(-3,(t0 + int((tt-t0)/n)),2)
        step_sizes = np.array([0.1])
        #step_sizes = np.linspace(0,(t0 + int((tt-t0)/n),5))
        #step_sizes = np.array([0.1,1])
        ODE = lokta
        # main(t0,tt,x0,y0,ODE, deltat_max, step_sizes)
        x_sols, y_sols = main(t0,tt,0.1,0.1,ODE,n, 0.5, step_sizes,a=1,b=b,d=0.1)
        plt.plot(x_sols, y_sols, label=b)

        plt.xlabel("t")
        plt.ylabel("x(t),y(t)")
        plt.legend()
        '''
        #sols.append(x_sol)
        #plt.scatter(bs,sols)
        plt.xlabel('Values of b')
        plt.ylabel('Prey(x(t_end))')
    '''
    plt.show()
>>>>>>> Stashed changes
