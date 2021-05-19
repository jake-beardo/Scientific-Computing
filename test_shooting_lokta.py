
import numpy as np
from shooting_functions import *


def lokta(t,x,a,d,b):
    u1 = x[0] * (1 - x[0]) - (x[1] * a * x[0]) / (d + x[0])
    u2 = b * x[1] * (1 - (x[1] / x[0]))
    return np.array([u1, u2])


a = 1
d = 0.1
b = 0.2

# us, period = shooting(np.array([1,1]),tt, hopf, 0.01, 'z', beta=beta, sigma=-1)
us, period = shooting(np.array([1,1]),200, lokta, 0.01, 'z', a=a, d=d,b=b)
print(us,period)
#
# plt.plot(t_vals, sols[:,0])
# plt.plot(t_vals, sols[:,1])
# plt.xlabel("t")
# plt.ylabel("x(t),y(t)")
# plt.show()
# theta = 20
# u_1= np.sqrt(beta)*np.cos(theta)
# u_2 = np.sqrt(beta)*np.sin(theta)
# tt = 0
# if abs(us[0] - np.sqrt(beta)*np.cos(tt+theta)) < 0.01 and abs(us[1] - np.sqrt(beta)*np.sin(tt+theta)) < 0.01:
#     print("Test passed")
#     print("Your output\nU0: ", us, '\nPeriod: ', period)
#     print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
# else:
#     print("Test failed")
#     print("Your output\nU0: ", us, '\nPeriod: ', period)
#     print("Expected output\nU0: ", u_1, u_2, '\nPeriod: ', period)
#
# u0 = np.array([1, 1])
# T = 20
