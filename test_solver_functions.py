''' TEST FILE '''

import unittest
import solver_functions
import numpy as np

class Test_solver_functions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setupClass')

    @classmethod
    def tearDownClass(cls):
        print('teardownClass')

    def setUp(self):
        print('setUp')
        def func1(t_pre,vars):
            return np.array([t_pre*(vars[0]+ np.sin(t_pre*np.pi)), vars[1]+np.sin(t_pre*np.pi)])
        def func2(vars, t_pre,a):
            return np.array(a*(vars + np.sin(t_pre)))
        def func3(vars,t_pre):
            return np.array([3*vars[0]+vars[1]-vars[2], vars[0]+2*vars[1]-vars[2], 3*vars[0] +3*vars[1]-vars[2]])
        def hopf(u_vals, t, beta, sigma):
            return np.array([beta*u_vals[0]-u_vals[1]+sigma*u_vals[0]*(u_vals[0]^2 + u_vals[1]^2),u_vals[0]+beta*u_vals[1]+sigma*u_vals[1]*(u_vals[0]^2 + u_vals[1]^2)])
        self.func1 = func1
        self.func2 = func2
        self.hopf = hopf

    ''' Solution to hopf '''
    # u_vals[0] = sqrt(beta)*np.cos(t+theta)
    # u_vals[1] = sqrt(beta)*np.sin(t+theta)
    # # where theta is the phase

    '''' Solution to func3 '''
# sol[0] = np.exp(2*t)*(3*x0 - z0)+np.exp(t)*(z*y0*np.sin(t)+(x0-z0)*(np.sin(t)-np.cos(t)))
# sol[1] = 2*np.exp(t)*(np.sin(t)*((x0+y0 - z0)+np.cos(t)*y0))
# sol[2] = np.exp(2*t)*(3*x0 - z0)+3*np.exp(t)*(2*y0*np.sin(t)+(x0+z0)*(np.sin(t)-np.cos(t)))

    def tearDown(self):
        print('tearDown\n')

    def test_euler_step(self):
        print('testing euler_step')
        result = solver_functions.euler_step(np.array([1,1]), 0, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal_nulp(result,np.array([1,2])))
        result = solver_functions.euler_step(np.array([-1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal_nulp(result,np.array([-2,-2])))
        result = solver_functions.euler_step(np.array([1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal_nulp(result,np.array([2,-2])))

        result = solver_functions.euler_step(3, 1, self.func2, 0.1, a=2)
        self.assertAlmostEqual(result,3.228224)



    def test_rk4(self):
        print('testing rk4')
        result = solver_functions.rk4(3, 1, self.func2, 0.1, a=2)
        self.assertAlmostEqual(result,3.228224,places=1)
        result = solver_functions.rk4(np.array([1,1]), 0, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([2.125, 3.7916]),decimal=1))
        result = solver_functions.rk4(np.array([-1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([-6.625,-3.791]),decimal=1))
        result = solver_functions.rk4(np.array([1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([ 2.125, -3.79]),decimal=1))

# solve_to(vars,t0,ODE, t2,step_size,rk_e,**kwargs)
    def test_solve_to(self):
        print('testing solve_to')
        result = solver_functions.solve_to(3, 0, self.func2,2, 0.1,'--euler', a=2)
        self.assertAlmostEqual(result,4.959330227798291,places=1)
        result = solver_functions.solve_to(3, 0, self.func2,2, 0.1,'--runge', a=2)
        self.assertAlmostEqual(result,5.013236648050785,places=1)
        result = solver_functions.solve_to(np.array([1,1]), 0, self.func1, 1,0.01,'--euler')
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([2.06801386, 3.76586895]),decimal=1))
        result = solver_functions.solve_to(np.array([1,1]), 0, self.func1, 1,0.01,'--runge')
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([2.08286038, 3.79296003]),decimal=1))
        result = solver_functions.solve_to(np.array([-10,-11]), 10, self.func1, 12,0.001,'--runge')
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([-3.48300231*(10**11), -7.94330171*10]),decimal=-100))

    def test_solve_ode(self):
        print('testing solve_to')
        result = solver_functions.solve_to(3, 0, self.func2,2, 0.1,'--euler', a=2)
        self.assertAlmostEqual(result,4.959330227798291,places=1)
        result = solver_functions.solve_to(3, 0, self.func2,2, 0.1,'--runge', a=2)
        self.assertAlmostEqual(result,5.013236648050785,places=1)
        result = solver_functions.solve_to(np.array([1,1]), 0, self.func1, 1,0.01,'--euler')
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([2.06801386, 3.76586895]),decimal=1))
        result = solver_functions.solve_to(np.array([1,1]), 0, self.func1, 1,0.01,'--runge')
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([2.08286038, 3.79296003]),decimal=1))
        result = solver_functions.solve_to(np.array([-10,-11]), 10, self.func1, 12,0.001,'--runge')
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([-3.48300231*(10**11), -7.94330171*10]),decimal=-100))

        #solve_ode(vars,tt, ODE,step_size=0.01,n=500, rk_e='--runge', **kwargs)
if __name__ == '__main__':
    unittest.main()
