# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi

def pde_solver(u_I,lmbda,x,mx,mt,bound_conds=np.array([0,0]), method='forward euler'):
    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step
    b_jp1 = np.zeros(x.size)
    # Set initial condition
    # u_j[0] =  u_I(x[0]) + bound_conds[0]
    for i in range(0, mx+1):
        # u_I function
        u_j[i] = u_I(x[i])
    # u_j[-1] =  u_I(x[-1]) + bound_conds[1]
    # Solve the PDE: loop over all time points
    for j in range(0, mt):
        b_jp1[1] = bound_conds[0]
        b_jp1[-2] = bound_conds[1]
        if method == 'forward euler':
            A_diag = np.diag([1-2*lmbda]*(mx-1))+np.diag([lmbda]*(mx-2),-1) + np.diag([lmbda]*(mx-2),1)
            u_jp1[1:mx] = A_diag@u_j[1:mx] + lmbda*b_jp1[1:mx]

        print(A_diag)
        print(A_diag.shape)

        # Boundary conditions
        u_jp1[0] = bound_conds[0]
        u_jp1[mx] = bound_conds[1]

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j,x
