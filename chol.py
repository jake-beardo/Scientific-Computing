import numpy as np

def dx_dt(x,y,a,d):
  return x*(1-x) - ((a*x*y)/(d+x))

def dy_dt(x,y,b):
    return b*y*(1-y/x)


a=1
d=0.1
b=np.linspace(0.1,0.5,100)
t=np.linspace(0,10,100)

dy_dx = dy_dt(x,y,b)*(1/dx_dt(x,y,a,d))
print(dy_dx)

#t_vals_runge, x_sols_runge = solve_ode(1, 0,1, 10, dx_dt,0.1,0.1,"--runge")
