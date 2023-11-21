import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from sgp import SparseGaussianProcess
from kernels import SquaredExponentialKernel


class TestDuplicateHandling(unittest.TestCase):

    def setUp(self):
        self.kernel = SquaredExponentialKernel

    def test_hyperparameter_init(self):
        SGP = SparseGaussianProcess(1, self.kernel, init_noise=0.123, init_outputscale=2.345)
        assert (SGP.noise - 0.123) < 1e-14
        assert (SGP.outputscale - 2.345) < 1e-14

        SGP = SparseGaussianProcess(1, self.kernel, init_noise=56.78910, init_outputscale=9.876,
                                    noise_range=[2, 100], outputscale_range=[7, 15])
        assert (SGP.noise - 56.78910) < 1e-14
        assert (SGP.outputscale - 9.876) < 1e-14


if __name__ == '__main__':
    unittest.main()
