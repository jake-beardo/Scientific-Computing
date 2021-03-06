import numpy as np
from solver_functions import *

def dx_dt(vars,a=1,b=0.3,d=0.1):
    dx_dt = vars[0]*(1-vars[0]) - (a*vars[0]*vars[1])/(d+vars[0])
    return dx_dt

def dy_dt(vars,a=1,b=0.3,d=0.1):
    dy_dt = b*vars[1]*(1 - (vars[1]/vars[0]))
    return dy_dt


a = 1
d = 0.1
b = np.linspace(0.1,0.5,100)
b = 0.1
t0 = 0
tt = 400
n=tt-1
step_sizes =  np.logspace(-3,(t0 + int((tt-t0)/n)),8)
ODE = dx_dt, dy_dt
# main(t0,tt,x0,y0,ODE, deltat_max, step_sizes)
main(t0,tt,0.1,0.1,ODE, 0.5, step_sizes)
