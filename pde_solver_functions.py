# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import pylab as pl
from math import pi

def set_bcs(mx,lmbda,x,u_jp1,u_j,A_diag,b_jp1,type_bc,method):
    if type_bc == 'Neumann':
        bc_vec = np.zeros(mx+1) # making the boundary condition vector
        bc_vec[0] = -b_jp1[1]
        bc_vec[-1]= b_jp1[-2]
        if method == forward:
            u_jp1[0:mx+1] = A_diag@u_j[0:mx+1] + 2*lmbda*(x[1]-x[0])*bc_vec
        elif method == backward:
            np.linalg.solve(A_diag, u_j[0:mx+1] + 2*lmbda*(x[1]-x[0])*bc_vec)
        elif method == crank:
            A = A_diag[0]
            B = A_diag[1]
            u_jp1[0:mx+1] = np.linalg.solve(A, ((B @ u_j[0:mx+1]) + 2*lmbda*(x[1]-x[0])*bc_vec))
    else:
        u_jp1[0] = b_jp1[1]
        u_jp1[mx] = b_jp1[-2]
    return u_jp1


def forward(u_j,u_jp1,lmbda,x,mx,b_jp1, type_bc):
    method=forward
    if type_bc == 'Neumann':
        A_diag = np.diag([1-2*lmbda]*(mx+1))+np.diag([lmbda]*(mx),-1) + np.diag([lmbda]*(mx),1)
        A_diag[0,1] = 2*lmbda
        A_diag[-1,-2] = 2*lmbda
        u_jp1 = set_bcs(mx,lmbda,x,u_jp1,u_j,A_diag,b_jp1,type_bc,method)

    else:
        A_diag = np.diag([1-2*lmbda]*(mx-1))+np.diag([lmbda]*(mx-2),-1) + np.diag([lmbda]*(mx-2),1)
        if type_bc == 'periodic':
            A_diag[0,-1] = lmbda
            A_diag[-1,0] = lmbda
            b_jp1[-2] = b_jp1[1]
        u_jp1[1:mx] = A_diag@u_j[1:mx] + lmbda*b_jp1[1:mx]
        u_jp1 = set_bcs(mx,lmbda,x,u_jp1,u_j,A_diag,b_jp1,type_bc,method)

    return u_jp1

def backward(u_j,u_jp1,lmbda,x,mx,b_jp1, type_bc):
    method=backward
    if type_bc == 'Neumann':
        A_diag = np.diag([1+2*lmbda]*(mx+1))+np.diag([-lmbda]*(mx),-1) + np.diag([-lmbda]*(mx),1)
        A_diag[0,1] = -2*lmbda
        A_diag[-1,-2] = -2*lmbda
        u_jp1 = set_bcs(mx,lmbda,x,u_jp1,u_j,A_diag,b_jp1,type_bc,method)

    else:
        A_diag = np.diag([1+2*lmbda]*(mx-1))+np.diag([-lmbda]*(mx-2),-1) + np.diag([-lmbda]*(mx-2),1)
        if type_bc == 'periodic':
            A_diag[0,-1] = -lmbda
            A_diag[-1,0] = -lmbda
            b_jp1[-2] = b_jp1[1]
        u_jp1[1:mx] = np.linalg.solve(A_diag, u_j[1:mx] + lmbda*b_jp1[1:mx])
        u_jp1 = set_bcs(mx,lmbda,x,u_jp1,u_j,A_diag,b_jp1,type_bc,method)

    return u_jp1



def crank(u_j,u_jp1,lmbda,x,mx,b_jp1, type_bc):
    method=crank
    if type_bc == 'Neumann':
        A = np.diag([1+lmbda]*(mx+1))+np.diag([-lmbda/2]*(mx),-1) + np.diag([-lmbda/2]*(mx),1)
        B = np.diag([1-lmbda]*(mx+1))+np.diag([lmbda/2]*(mx),-1) + np.diag([lmbda/2]*(mx),1)
        A[0,1] = -2*lmbda
        A[-1,-2] = -2*lmbda
        B[0,1] = 2*lmbda
        B[-1,-2] = 2*lmbda
        '''change set bcs to do Neumann'''
        u_jp1 = set_bcs(mx,lmbda,x,u_jp1,u_j,[A,B],b_jp1,type_bc,method)

    else:
        A = np.diag([1+lmbda]*(mx-1))+np.diag([-lmbda/2]*(mx-2),-1) + np.diag([-lmbda/2]*(mx-2),1)
        B = np.diag([1-lmbda]*(mx-1))+np.diag([lmbda/2]*(mx-2),-1) + np.diag([lmbda/2]*(mx-2),1)
        if type_bc == 'periodic':
            A[0,-1] = -lmbda
            A[-1,0] = -lmbda
            B[0,-1] = lmbda
            B[-1,0] = lmbda
            b_jp1[-2] = b_jp1[1]

        u_jp1[1:mx] = np.linalg.solve(A, ((B @ u_j[1:mx]) + lmbda*b_jp1[1:mx]))
        u_jp1 = set_bcs(mx,lmbda,x,u_jp1,u_j,[A,B],b_jp1,type_bc,method)
    return u_jp1




def pde_solver(u_I,lmbda,x,mx,mt,bound_conds, method=forward, type_bc='Dirichlet'):
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
        b_jp1[1] = bound_conds[0](j)
        b_jp1[-2] = bound_conds[-1](j)
        u_jp1 = method(u_j,u_jp1,lmbda, x, mx, b_jp1, type_bc)

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j,x
