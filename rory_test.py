"""
Tests for students for the Hybrid images (PA1) assignment
Convention: append an integer to the end of the test, for multiple versions of
the same test at different difficulties.  Higher numbers are more difficult
(lower thresholds or accept fewer mistakes).  Example:
    test_all_equal1(self):
    ...
    test_all_equal2(self):
    ...
"""
import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')


import unittest

import cv2
import numpy as np

import hybrid

class TestGaussianKernel2D(unittest.TestCase):
    def test_5_5_5(self):
        a = np.array([[ 0.03688345,  0.03916419,  0.03995536,  0.03916419,  0.03688345],
            [ 0.03916419,  0.04158597,  0.04242606,  0.04158597,  0.03916419],
            [ 0.03995536,  0.04242606,  0.04328312,  0.04242606,  0.03995536],
            [ 0.03916419,  0.04158597,  0.04242606,  0.04158597,  0.03916419],
            [ 0.03688345,  0.03916419,  0.03995536,  0.03916419,  0.03688345]])

        # alternate result, which is based on more exact numeric integral
        a_alternate = np.array([[0.03689354, 0.03916709, 0.03995566, 0.03916709, 0.03689354],
                         [0.03916709, 0.04158074, 0.0424179,  0.04158074, 0.03916709],
                         [0.03995566, 0.0424179,  0.04327192, 0.0424179,  0.03995566],
                         [0.03916709, 0.04158074, 0.0424179,  0.04158074, 0.03916709],
                         [0.03689354, 0.03916709, 0.03995566, 0.03916709, 0.03689354]])
        self.assertTrue(np.allclose(hybrid.gaussian_blur_kernel_2d(5, 5, 5), a, rtol=1e-4, atol=1e-08) 
            or np.allclose(hybrid.gaussian_blur_kernel_2d(5, 5, 5), a_alternate, rtol=1e-4, atol=1e-08))

    def test_1_7_3(self):
        a = np.array([[ 0.00121496,  0.00200313,  0.00121496],
            [ 0.01480124,  0.02440311,  0.01480124],
            [ 0.06633454,  0.10936716,  0.06633454],
            [ 0.10936716,  0.18031596,  0.10936716],
            [ 0.06633454,  0.10936716,  0.06633454],
            [ 0.01480124,  0.02440311,  0.01480124],
            [ 0.00121496,  0.00200313,  0.00121496]])

        # alternate result, which is based on more exact numeric integral
        a_alternate = np.array([[0.00166843, 0.00264296, 0.00166843],
            [0.01691519, 0.02679535, 0.01691519],
            [0.0674766,  0.10688965, 0.0674766 ],
            [0.10688965, 0.16932386, 0.10688965],
            [0.0674766,  0.10688965, 0.0674766 ],
            [0.01691519, 0.02679535, 0.01691519],
            [0.00166843, 0.00264296, 0.00166843]])
        self.assertTrue(np.allclose(hybrid.gaussian_blur_kernel_2d(1, 7, 3), a, rtol=1e-4, atol=1e-08)
            or np.allclose(hybrid.gaussian_blur_kernel_2d(1, 7, 3), a_alternate, rtol=1e-4, atol=1e-08))

    def test_1079_3_5(self):
        a = np.array([[ 0.06600011,  0.06685595,  0.06714369,  0.06685595,  0.06600011],
            [ 0.06628417,  0.06714369,  0.06743267,  0.06714369,  0.06628417],
            [ 0.06600011,  0.06685595,  0.06714369,  0.06685595,  0.06600011]])

        # alternate result, which is based on more exact numeric integral
        a_alternate = np.array([[0.06600058, 0.06685582, 0.06714335, 0.06685582, 0.06600058],
             [0.06628444, 0.06714335, 0.06743212, 0.06714335, 0.06628444],
             [0.06600058, 0.06685582, 0.06714335, 0.06685582, 0.06600058]])
        self.assertTrue(np.allclose(hybrid.gaussian_blur_kernel_2d(10.79, 3, 5), a, rtol=1e-4, atol=1e-08)
            or np.allclose(hybrid.gaussian_blur_kernel_2d(10.79, 3, 5), a_alternate, rtol=1e-4, atol=1e-08))

if __name__ == '__main__':
    np.random.seed(4670)
    unittest.main()
