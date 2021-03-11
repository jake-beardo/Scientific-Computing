import numpy as np
from solver_functions import *

def lokta(vars,t,a,b,d):
    return np.array([vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0]), b*vars[1]*(1 - (vars[1]/vars[0]))])


if __name__ == "__main__":

    bs = [0.1,0.2,0.3,0.4,0.5]
    for b in bs:
        a = 1
        d = 0.1
        print(b,'b...')

        #b = np.linspace(0.1,0.5,100)
        #b = 0.1
        t0 = 0
        tt = 1000
        n = 10000
        step_sizes =  np.logspace(-3,(t0 + int((tt-t0)/n)),2)
        ODE = lokta
        # main(t0,tt,x0,y0,ODE, deltat_max, step_sizes)
        main(t0,tt,1,0.5,ODE,n, 0.5, step_sizes,a=1,b=b,d=0.1)

    plt.show()
