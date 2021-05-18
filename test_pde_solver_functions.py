''' test forward euler '''

import numpy as np
import pylab as pl
from math import pi
import matplotlib.pyplot as plt
from pde_solver_functions import pde_solver, forward, backward, crank

# Set problem parameters/functions
kappa = 0.1   # diffusion constant
L=1.0         # length of spatial domain
T=0.5         # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 10     # number of gridpoints in space
mt = 1000   # number of gridpoints in time



# Set up the numerical environment variables
x = np.linspace(0, L, mx+1)     # mesh points in space
t = np.linspace(0, T, mt+1)     # mesh points in time
deltax = x[1] - x[0]            # gridspacing in x
deltat = t[1] - t[0]            # gridspacing in t
lmbda = kappa*deltat/(deltax**2)    # mesh fourier number
print("deltax=",deltax)
print("deltat=",deltat)
print("lambda=",lmbda)


# These are the boundary condition functions
def p_func(t):
    return t

def q_func(t):
    return 0


method=backward
method=crank
method=crank
type_bc = 'periodic'
bound_conds = np.array([p_func,q_func])
u_j,x,steady_state = pde_solver(u_I,lmbda,x,mx,mt,bound_conds,method,type_bc)

# Plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
# pl.legend(loc='upper right')
pl.show()
