import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from utils import cholesky_rankone_update, cholesky_update


class TestUtils(unittest.TestCase):

    def test_chol_updates(self):
        print('Testing Cholesky updates ...')

        def test_rankone(N):
            A = torch.rand((N, N), dtype=torch.float64)
            A = A.T @ A + 1e-8 * torch.eye(A.shape[0])  # make positive definite
            L = torch.linalg.cholesky(A)
            v = torch.rand(N)
            newL = cholesky_rankone_update(L, v, 1)
            trueL = torch.linalg.cholesky(A + torch.outer(v, v))
            assert torch.max(torch.abs((newL - trueL))) < 1e-4

        test_rankone(10)
        test_rankone(100)
        test_rankone(1000)

        def test_rankN(N, M):
            A = torch.rand((N, N), dtype=torch.float64)
            A = A.T @ A + 1e-8 * torch.eye(A.shape[0])  # make positive definite
            L = torch.linalg.cholesky(A)
            V = torch.rand(N, M)
            newL = cholesky_update(L, V, 1)
            trueL = torch.linalg.cholesky(A + V @ V.T)
            assert torch.max(torch.abs((newL - trueL))) < 1e-4

        test_rankN(10, 5)
        test_rankN(100, 50)
        test_rankN(1000, 50)
        print('Cholesky updates work as expected')


if __name__ == '__main__':
    unittest.main()
