''' test forward euler '''

import numpy as np
import pylab as pl
from math import pi
import matplotlib.pyplot as plt
from pde_solver_functions import pde_solver, forward, backward, crank

# Set problem parameters/functions
kappa = 0.5   # diffusion constant
L=1.0         # length of spatial domain
T=1        # total time to solve for
def u_I(x):
    # initial temperature distribution
    y = np.sin(pi*x/L)
    return y

def u_exact(x,t):
    # the exact solution
    y = np.exp(-kappa*(pi**2/L**2)*t)*np.sin(pi*x/L)
    return y

# Set numerical parameters
mx = 30    # number of gridpoints in space
mt = 30  # number of gridpoints in time

# These are the boundary condition functions
def p_func(t):
    return 1

def q_func(t):
    return 4


method=backward
method=crank
method=backward
type_bc = 'Dirichlet'
bound_conds = np.array([p_func,q_func])

# u_I,L,T,main_param, ,mx,mt,bound_conds,varied_param=main_param, method=forward, type_bc='Dirichlet', **kwargs
u_j,x,steady_state = pde_solver(u_I,L,T,mx,mt,bound_conds,main_param='kappa',method=method,type_bc=type_bc,kappa=kappa)
print(steady_state)

# Plot the final result and exact solution
pl.plot(x,u_j,'ro',label='num')
xx = np.linspace(0,L,250)
pl.plot(xx,u_exact(xx,T),'b-',label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
# pl.legend(loc='upper right')
pl.show()
