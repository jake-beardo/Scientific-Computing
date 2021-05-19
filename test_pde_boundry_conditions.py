import numpy as np
import pylab as pl
from math import pi
import matplotlib.pyplot as plt
from pde_solver_functions import pde_solver, forward, backward, crank
from continuation_functions import continuation_natural

param={'kappa': 1,  # diffusion constant
            'L': 1,     # length of spatial domain
            'T': 0.5}   # total time to solve for
neumann_boundary_conditions, 'periodic',
def neumann_boundary_conditions(t): # gradient at boundaries known
    dudx_zerot = np.sin(t)
    dudx_Lt = 0
    return np.array([dudx_zerot, dudx_Lt])
