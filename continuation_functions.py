import numpy as np
from scipy.optimize  import fsolve
from solver_functions import solve_ode, solve_to, euler_step, rk4
from shooting_functions import shooting_main, shooting, integrate, get_phase_conditon,period_finder
from main import func
from matplotlib import pyplot as plt
