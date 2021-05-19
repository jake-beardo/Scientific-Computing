''' simple test for PDE continuation'''
import numpy as np
import pylab as pl
from math import pi
import matplotlib.pyplot as plt
from pde_solver_functions import pde_solver, forward, backward, crank
from continuation_functions import continuation_natural

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
def p_func(t,a):
    return a

def q_func(t):
    return 4

method=backward
method=crank
method=crank
type_bc = 'Dirichlet'
bound_conds = np.array([p_func,q_func])

# continuation_natural(init_guess, tt, ODE, init_param, discretisation=False, param_step_size=0.1, param_from=0,param_to=2, step_size=0.01,n=500, rk_e='--runge',bound_conds=np.array([0,0]),method=forward,type_bc='Dirichlet', **kwargs):
steady_states,params = continuation_natural(1.6, mt, u_I, 'a', discretisation=pde_solver, param_step_size=0.05, param_from=0.4,param_to=2,main_param='kappa',L=L,T=T,mx=mx,bound_conds=bound_conds,method=method,type_bc=type_bc, kappa=kappa,a=1)

# Plot the final result and exact solution
pl.plot(params,steady_states,'r-',label='num')

pl.xlabel('x')
pl.ylabel('u(x,0.5)')
# pl.legend(loc='upper right')
pl.show()

# PDE continuation
