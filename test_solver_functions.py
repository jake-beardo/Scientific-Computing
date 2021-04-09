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
        self.func1 = func1
        self.func2 = func2

    def tearDown(self):
        print('tearDown\n')

    def test_euler_step(self):
        print('test_euler_step')
        result = solver_functions.euler_step(np.array([1,1]), 0, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal_nulp(result,np.array([1,2])))
        result = solver_functions.euler_step(np.array([-1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal_nulp(result,np.array([-2,-2])))
        result = solver_functions.euler_step(np.array([1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal_nulp(result,np.array([2,-2])))

        result = solver_functions.euler_step(3, 1, self.func2, 0.1, a=2)
        self.assertAlmostEqual(result,3.228224)



    def test_rk4(self):
        print('test_rk4')
        result = solver_functions.rk4(3, 1, self.func2, 0.1, a=2)
        self.assertAlmostEqual(result,3.228224,places=1)
        result = solver_functions.rk4(np.array([1,1]), 0, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([2.125, 3.7916]),decimal=1))
        result = solver_functions.rk4(np.array([-1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([-6.625,-3.791]),decimal=1))
        result = solver_functions.rk4(np.array([1,-1]), 1, self.func1, 1)
        self.assertIsNone(np.testing.assert_array_almost_equal(result,np.array([ 2.125, -3.79]),decimal=1))



if __name__ == '__main__':
    unittest.main()
