import torch
import unittest

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../source'))
from sgp import SparseGaussianProcess


class TestDuplicateHandling(unittest.TestCase):

    def setUp(self):
        self.SGP = SparseGaussianProcess(1)

    def test(self):
        x1 = torch.tensor([1, 2, 3, 4, 5, 8], dtype=torch.float64)
        x2 = torch.tensor([1 - 1e-8, 2.25, 2.5, 3, 4 + 1e-9, 5.5, 6, 7, 8 + 1e-8],
                          dtype=torch.float64)

        duplicate_ids = self.SGP.get_duplicate_ids(torch.atleast_2d(x1),
                                                   torch.atleast_2d(x2),
                                                   tol=1e-8)
        self.assertTrue(torch.equal(duplicate_ids, torch.tensor([0, 3, 4, 8])))
        x2_clean = self.SGP.remove_duplicates(torch.atleast_2d(x1), torch.atleast_2d(x2), tol=1e-8)
        self.assertTrue(torch.equal(x2_clean, torch.atleast_2d(torch.tensor([2.25, 2.5, 5.5, 6, 7],
                                                                            dtype=torch.float64))))

        duplicate_ids = self.SGP.get_duplicate_ids(torch.atleast_2d(x2),
                                                   torch.atleast_2d(x1),
                                                   tol=1e-8)
        self.assertTrue(torch.equal(duplicate_ids, torch.tensor([0, 2, 3, 5])))
        x1_clean = self.SGP.remove_duplicates(torch.atleast_2d(x2), torch.atleast_2d(x1), tol=1e-8)
        self.assertTrue(torch.equal(x1_clean, torch.atleast_2d(torch.tensor([2, 5], dtype=torch.float64))))

        x3 = torch.tensor([1, 2, 3, 1 - 1e-9, 4, 5], dtype=torch.float64)
        duplicate_ids = self.SGP.get_duplicate_ids(torch.atleast_2d(x3),
                                                   torch.atleast_2d(x3),
                                                   tol=1e-8)
        self.assertTrue(torch.equal(duplicate_ids, torch.tensor([3])))
        x3_clean = self.SGP.remove_duplicates(torch.atleast_2d(x3), torch.atleast_2d(x3), tol=1e-8)
        self.assertTrue(torch.equal(x3_clean, torch.atleast_2d(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64))))


if __name__ == '__main__':
    unittest.main()
