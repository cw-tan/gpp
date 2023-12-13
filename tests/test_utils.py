import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from utils import cholesky_update


class TestUtils(unittest.TestCase):

    def test_chol_updates(self):
        print('Testing Cholesky updates ...')

        def test_rankN(N, M):
            A = torch.rand((N, N), dtype=torch.float64)
            A = A.T @ A + 1e-5 * torch.eye(A.shape[0], dtype=A.dtype)  # make positive definite
            L = torch.linalg.cholesky(A)
            V = torch.rand(N, M, dtype=torch.float64)
            A_prime = A + V @ V.T
            L = cholesky_update(L, V)
            assert torch.max(torch.abs((A_prime - L @ L.T))) <= 1e-10  # for float64

        test_rankN(10, 2)
        test_rankN(100, 20)
        test_rankN(1000, 200)
        print('Cholesky updates work as expected')


if __name__ == '__main__':
    unittest.main()
