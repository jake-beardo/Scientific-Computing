import numpy as np
from solver_functions import *

def dx_dt(x,y,a,b,d):
  dx_dt = x*(1-x) - (a*x*y)/(d+x)
  return dx_dt

 def dy_dt(x,y,a,b,d):
   dy_dt = b*y*(1-(y/x))
   return dy_dt


  a = 1
  d = 0.1
  b = np.linspace(0.1,0.5,100)
  b = 0.1


  step_sizes =  np.logspace(-3,(t0 + int((tt-t0)/n)),8)
ODE = dx_dt, dy_dt
  main(0,1,1,1,ODE, 0.5, step_sizes)
