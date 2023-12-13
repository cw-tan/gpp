import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from sgp import SparseGaussianProcess
from kernels import SquaredExponentialKernel


def func(x):
    return torch.cos(0.5 * x) - 0.3 * x + 0.1 * torch.exp(0.3 * x) + 0.5 * torch.rand(x.shape, dtype=torch.float64)


class TestSGPOutputs(unittest.TestCase):

    def setUp(self):
        self.kernel = SquaredExponentialKernel()
        L = 10
        x_train = L * (2 * torch.rand(1000, dtype=torch.float64) - 1)
        self.y_train = func(x_train)
        self.x_train = torch.atleast_2d(x_train)
        self.x_sparse = torch.atleast_2d(x_train[torch.randperm(len(x_train))[:200]])
        self.x_test = torch.atleast_2d(torch.linspace(-L - 1, L + 1, 1000, dtype=torch.float64))

    def test_updates(self):
        print('Testing efficient updates ...')
        for mode in ['sor', 'dtc', 'fitc', 'vfe']:
            for decomp in ['v', 'qr']:
                SGP1 = SparseGaussianProcess(1, self.kernel, decomp_mode=decomp, sgp_mode=mode)
                SGP1.update_model(self.x_train, self.y_train, self.x_sparse)

                SGP2 = SparseGaussianProcess(1, self.kernel, decomp_mode=decomp, sgp_mode=mode)
                SGP2.update_model(self.x_train[:, :500], self.y_train[:500], self.x_sparse[:, :100])
                SGP2.update_model(self.x_train[:, 500:750], self.y_train[500:750], None)
                SGP2.update_model(None, None, self.x_sparse[:, 100:150])
                SGP2.update_model(self.x_train[:, 750:], self.y_train[750:], self.x_sparse[:, 150:])

                assert torch.equal(SGP1.full_descriptors, SGP2.full_descriptors)
                assert torch.equal(SGP1.full_descriptors, SGP2.full_descriptors)
                assert torch.equal(SGP1.full_descriptors, SGP2.full_descriptors)

                mean1, var1 = SGP1(self.x_test, mean_var=[True, True], include_noise=False)
                mean2, var2 = SGP2(self.x_test, mean_var=[True, True], include_noise=False)

                #print(torch.mean(torch.abs(mean1 - mean2)).item())
                #print(torch.max(torch.abs(mean1 - mean2)).item())
                #print(torch.mean(torch.abs(var1 - var2)).item())
                #print(torch.max(torch.abs(var1 - var2)).item())
                assert torch.max(torch.abs(mean1 - mean2)) < 5e-3
                assert torch.max(torch.abs(var1 - var2)) < 5e-3
        print('Efficient updates work as expected')


if __name__ == '__main__':
    unittest.main()
