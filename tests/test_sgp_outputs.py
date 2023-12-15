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
        x_train = L * (2 * torch.rand(500, dtype=torch.float64) - 1)
        self.y_train = func(x_train)
        self.x_train = torch.atleast_2d(x_train).T
        self.x_sparse = self.kernel.remove_duplicates(self.x_train, self.x_train, tol=1e-1)
        # ^ very important to prune inducing points close together for matrices to be
        #   well conditioned and reduce error during triangular solves
        self.x_test = torch.atleast_2d(L * (2 * torch.rand(1000, dtype=torch.float64) - 1)).T

    def test_updates(self):
        Ns = self.x_sparse.shape[1]
        for mode in ['sor', 'dtc', 'fitc', 'vfe']:
            for decomp in ['v', 'qr']:
                # load all data at once
                SGP1 = SparseGaussianProcess(1, self.kernel, decomp_mode=decomp, sgp_mode=mode)
                SGP1.update_model(self.x_train, self.y_train, self.x_sparse)

                # load data sequentially
                SGP2 = SparseGaussianProcess(1, self.kernel, decomp_mode=decomp, sgp_mode=mode)
                SGP2.update_model(self.x_train[:500, :], self.y_train[:500], self.x_sparse[:int(Ns / 2), :])
                SGP2.update_model(self.x_train[500:750, :], self.y_train[500:750], None)
                SGP2.update_model(None, None, self.x_sparse[int(Ns / 2):int(3 / 4 * Ns), :])
                SGP2.update_model(self.x_train[750:, :], self.y_train[750:], self.x_sparse[int(3 / 4 * Ns):, :])

                # check if all data loaded is identical
                assert torch.equal(SGP1.full_descriptors, SGP2.full_descriptors)
                assert torch.equal(SGP1.training_outputs, SGP2.training_outputs)
                assert torch.equal(SGP1.sparse_descriptors, SGP2.sparse_descriptors)

                mean1, var1 = SGP1(self.x_test, mean_var=[True, True], include_noise=False)
                mean2, var2 = SGP2(self.x_test, mean_var=[True, True], include_noise=False)

                # check closeness of intermediates matrices and predictions
                assert torch.max(torch.abs(SGP1.Lss @ SGP1.Lss.T - SGP2.Lss @ SGP2.Lss.T)) < 1e-14
                assert torch.max(torch.abs(SGP1.L_Sigma @ SGP1.L_Sigma.T - SGP2.L_Sigma @ SGP2.L_Sigma.T)) < 1e-8
                assert torch.max(torch.abs(SGP1.alpha - SGP2.alpha)) < 1e-7, \
                    'value is {}'.format(torch.max(torch.abs(SGP1.alpha - SGP2.alpha)))
                assert torch.max(torch.abs(mean1 - mean2)) < 1e-8, \
                    'value is {}'.format(torch.max(torch.abs(mean1 - mean2)))
                assert torch.max(torch.abs(var1 - var2)) < 1e-8, \
                    'value is {}'.format(torch.max(torch.abs(var1 - var2)))

        print('Efficient updates work as expected')


if __name__ == '__main__':
    unittest.main()
