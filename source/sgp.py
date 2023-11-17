import numpy as np
import torch
from linear_operator.operators import IdentityLinearOperator, DiagLinearOperator, LinearOperator, TriangularLinearOperator, ConstantDiagLinearOperator, CholLinearOperator, RootLinearOperator
from linear_operator import settings
from linear_operator import root_decomposition, root_inv_decomposition
from linear_operator.operators import to_linear_operator

import time
from linear_operator.utils import stable_qr

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lbfgs'))

from lbfgsnew import LBFGSNew

# Notes:
# 1. linear operator Cholesky decomposition seems slower than native torch on cpu

softplus = torch.nn.Softplus()

# TODOs: 
# 1. make kernel a model child and have it carry its own hyperparameters
# 2. implement low-rank update of U @ U.T decomposition
# 3. hyperparameter optimization based on KL divergence/ELBO

def kernel(x1, x2, noise=1, l=1, diag=False):
      """
      x1 (torch.Tensor) : d × m tensor where m is the number of x vectors and d is the
                          dimensionality of the x vectors
      x2 (torch.Tensor) : d × n tensor n where is the number of x vectors and d is the
                          dimensionality of the x vectors
      """
      assert torch.is_tensor(x1) and torch.is_tensor(x2), 'x1 and x2 must be torch.Tensor'
      assert 0 < len(x1.shape) <= 2 and 0 < len(x2.shape) <= 2, 'x1 and x2 must be 1D or 2D'
      x1_mat = torch.atleast_2d(x1)
      x2_mat = torch.atleast_2d(x2)
      assert x1_mat.shape[0] == x2_mat.shape[0], 'vector dimensions of x1 and x2 are incompatible'

      if not diag:
          x1_mat = torch.transpose(x1_mat.unsqueeze(2).expand(x1_mat.shape + (x2_mat.shape[1],)), 0, 1)  # [m, d, n]
      else:
          assert x1_mat.shape[1] == x2_mat.shape[1], 'diag = True requires same dims'
      # the only thing different for other kernels is the last two lines
      scalar_form = torch.sum((x1_mat - x2_mat).pow(2), dim=1)  # [m, n]
      # for stability during hyperparameter optimization
      noise = softplus(noise) + 1e-7
      return noise * noise * torch.exp(-0.5 * scalar_form / (l * l))


class SparseGaussianProcess():
    """
    TODO: device manipulation
    """
    def __init__(self, descriptor_dim, invert_mode='QR', variance_mode='dtc'):
        """
        
        """
        self.descriptor_dim = descriptor_dim
        self.full_descriptors = torch.empty((descriptor_dim, 0))
        self.sparse_descriptors = torch.empty((descriptor_dim, 0))
        self.training_outputs = torch.empty((0,))

        # hyperparameters
        self.kernel_noise = torch.tensor([1], dtype=torch.float64)
        self.kernel_length = torch.tensor([1], dtype=torch.float64)
        self.kernel_hyperparameters = torch.tensor([1, 1], dtype=torch.float64)
        self.model_noise = torch.tensor([0.01], dtype=torch.float64)

        self.optimizer = LBFGSNew([self.model_noise, self.kernel_noise, self.kernel_length], 
                                   lr=1e-2, history_size=8, max_iter=6)

        self.invert_mode = invert_mode
        self.variance_mode = variance_mode
        # trained model parameters (precomputed during model updates for faster prediction)
        self.Ksf = None
        self.Lambda_inv = None
        self.Kss = None
        self.Sigma = None
        self.alpha = None  # for predicting mean efficiently (just a matmul with this)

    def update_model(self, x_train, y_train, x_sparse):
        """
        Updates GP model with a set of training vectors, i.e. update the Cholesky decomposed L

        Args:
          x_train (torch.Tensor): d × m tensor where m is the number of x vectors and d is the
                                  dimensionality of the x descriptor vectors
          y_train (torch.Tensor): m-dimensional vector of training outputs corresponding to
                                  the training inputs x_train
        """
        assert torch.atleast_2d(x_train).shape[0] == self.descriptor_dim,\
          'x_train\'s dimensions are incompatible with the descriptor dimensions of the model'
        self.sparse_descriptors = torch.cat((self.sparse_descriptors, torch.atleast_2d(x_sparse)), dim=1)
        self.full_descriptors = torch.cat((self.full_descriptors, torch.atleast_2d(x_train)), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))
        self.__update_Sigma_alpha()

    def __update_Sigma_alpha(self):

        self.Ksf = kernel(self.sparse_descriptors, self.full_descriptors,
                          self.kernel_noise, self.kernel_length)

        # compute Kss
        Kss = kernel(self.sparse_descriptors, self.sparse_descriptors,
                     self.kernel_noise, self.kernel_length)
        Kss = Kss + torch.eye(Kss.shape[0]) * 1e-8  # jitter
        Lss = torch.linalg.cholesky(Kss)
        self.Kss = CholLinearOperator(Lss, upper=False)
 
        self.Lambda_inv = self.__get_Lambda_inv(self.full_descriptors)
 
        invert_start = time.time()
        match self.invert_mode:
            case 'N':  # use 'Normal Equations' method, i.e. direct Cholesky inverse (very unstable)
                Sigma_inv = Kss + self.Ksf @ self.Lambda_inv @ self.Ksf.T  # O(M²N)
                Sigma_inv = Sigma_inv + torch.eye(self.sparse_descriptors.shape[1]) * 1  # large jitter!
                L_Sigma_inv = torch.linalg.cholesky(Sigma_inv)  # O(M³)
                self.U_Sigma = TriangularLinearOperator(L_Sigma_inv, upper=False).inverse().T  # O(M³)
                #print('N method | cond(L_Σ_inv) = {:.4g}'.format(torch.linalg.cond(L_Sigma_inv).item()))
            case 'V':  # usually stable, sometimes breaks during hyperparameter optimization
                Lss_inv = self.Kss.cholesky().inverse()  # O(M³)
                V = self.Ksf.T @ Lss_inv.T  # O(M²N)
                Lambda_inv_sqrt_V = self.Lambda_inv.sqrt() @ V  # O(MN) since Lambda_inv is diagonal
                Gamma = IdentityLinearOperator(V.shape[1]) + Lambda_inv_sqrt_V.T @ Lambda_inv_sqrt_V  # O(M²N)
                L_Gamma = torch.linalg.cholesky(Gamma.to_dense())  # O(M³)
                U_Sigma = (TriangularLinearOperator(L_Gamma, upper=False).inverse() @ Lss_inv).T  # O(M³)
                self.Sigma = RootLinearOperator(U_Sigma)
                #print('V method | cond(L_Γ) = {:.4g}'.format(torch.linalg.cond(L_Gamma).item()))
            case 'QR':  # very stable
                B = torch.cat([self.Lambda_inv.sqrt() @ self.Ksf.T, Lss.T], dim=0)
                Q, R = stable_qr(B)
                U_Sigma = TriangularLinearOperator(R, upper=True).inverse()  # O(M³)
                self.Sigma = RootLinearOperator(U_Sigma)
        invert_end = time.time()
        #print('{} method | Total inversion time: {:.5g}s'.format(self.sigma_mode, invert_end - invert_start))

        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N²)
        self.alpha = self.Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)
 
    def update_full_set(self, x_train, y_train):
        """
        Update full set in a way that scales cubically in the number
        of full set points added rather than cubically in the number
        of sparse set points possessed.
        """
        self.full_descriptors = torch.cat((self.full_descriptors, torch.atleast_2d(x_train)), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))

        Ksfprime = kernel(self.sparse_descriptors, torch.atleast_2d(x_train),
                          self.kernel_noise, self.kernel_length)
        self.Ksf = torch.cat([self.Ksf, Ksfprime], dim=1)

        # Update Σ (or U_Σ)
        Lambda_prime = self.__get_Lambda_inv(torch.atleast_2d(x_train), update=True).inverse()
        to_invert = Lambda_prime + Ksfprime.T @ self.Sigma @ Ksfprime
        L_to_invert = torch.linalg.cholesky(to_invert.to_dense())  # O(N'³)

        C = TriangularLinearOperator(L_to_invert).solve(Ksfprime.T @ self.Sigma.root)
        I_minus_CCT = IdentityLinearOperator(C.shape[1]) - C.T @ C  # O(M²N')
        # TODO: the following scales as O(M³)
        # solution: custom rank-1 update (scales as O(M²)) 
        #           applied N' times, overall # O(M²N'), i.e.
        # B = TriangularLinearOperator(L_to_invert).solve(Ksfprime.T @ self.Sigma).T
        # self.Sigma = lowrank_update_UUT(self.Sigma, B)
        Uprime = root_decomposition(I_minus_CCT, method='cholesky').root  # O(M³)
        self.Sigma = RootLinearOperator(self.Sigma.root @ Uprime)  # O(M³) matmul

        self.Lambda_inv = DiagLinearOperator(torch.cat([self.Lambda_inv.diagonal(),
                                                        Lambda_prime.inverse().diagonal()]))
        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N²)
        self.alpha = self.Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

    def update_sparse_set(self, x_sparse):
        """
        Update with M' new sparse points 
        """
        Kssprime = kernel(self.sparse_descriptors, torch.atleast_2d(x_sparse),
                          self.kernel_noise, self.kernel_length)
        self.sparse_descriptors = torch.cat((self.sparse_descriptors, torch.atleast_2d(x_sparse)), dim=1)
        Ksprimesprime = kernel(torch.atleast_2d(x_sparse), torch.atleast_2d(x_sparse),
                               self.kernel_noise, self.kernel_length)
        Kfsprime = kernel(self.full_descriptors, torch.atleast_2d(x_sparse),
                          self.kernel_noise, self.kernel_length)

        # update Kss
        Lss_inv_Kssp = self.Kss.cholesky().solve(Kssprime)  # O(M²M')
        Lsp = torch.linalg.cholesky(Ksprimesprime - Lss_inv_Kssp.T @ Lss_inv_Kssp + torch.eye(Ksprimesprime.shape[0]) * 1e-8)  # O(M'³)
        newLss_upper = torch.cat([self.Kss.cholesky().to_dense(), torch.zeros(Kssprime.shape)], dim=1)
        newLss_lower = torch.cat([Lss_inv_Kssp.T, Lsp.to_dense()], dim=1)
        newLss = torch.cat([newLss_upper, newLss_lower], dim=0)
        self.Kss = CholLinearOperator(newLss, upper=False)

        # Update Σ (or U_Σ)
        Lambda_inv_Ksfp = self.Lambda_inv @ Kfsprime  # O(NM') since Lambda_inv is diagonal
        B = Kssprime + self.Ksf @ Lambda_inv_Ksfp  # O(NMM')
        Lambda_inv_root_Ksfp = self.Lambda_inv.sqrt() @ Kfsprime  # O(NM') since Lambda_inv is diagonal
        C = Ksprimesprime + Lambda_inv_root_Ksfp.T @ Lambda_inv_root_Ksfp  # O(N²M')
        aux = B.T @ self.Sigma.root  # O(M²M')
        to_invert = C - aux @ aux.T  # O(MM'²)
        U_Psi = root_inv_decomposition(to_invert, method='symeig').root  # O(M'³)
        upper_right = B @ U_Psi  # O(MM'²)
        upper_right = -1 * self.Sigma @ upper_right  # O(M²M')
        U_Sigma_upper = torch.cat([self.Sigma.root.to_dense(), upper_right], dim=1)
        U_Sigma_lower = torch.cat([torch.zeros((U_Psi.shape[0], self.Sigma.root.shape[1])), U_Psi.to_dense()], dim=1)
        U_Sigma = torch.cat([U_Sigma_upper, U_Sigma_lower], dim=0)
        self.Sigma = RootLinearOperator(U_Sigma)

        self.Ksf = torch.cat([self.Ksf, Kfsprime.T], dim=0)

        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N²)
        self.alpha = self.Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

    def get_predictions(self, x_test, mean_var=[True, True]):
        """
        Get predictions of GP model with a set of testing vectors.

        Args:
          x_test (torch.Tensor): d × p tensor where p is the number of x vectors and d is the
                                  dimensionality of the x descriptor vectors
        """
        Kst = kernel(self.sparse_descriptors, x_test,
                     self.kernel_noise, self.kernel_length)
        predictions = []
        if mean_var[0]:
            mean = Kst.T @ self.alpha
            predictions.append(mean)
        if mean_var[1]:
            match self.variance_mode:
                case 'sor':  # quite nonsensical
                    U_Sigma_Kst = self.Sigma.root.T @ Kst
                    var = U_Sigma_Kst.pow(2).sum(dim=0)
                case 'dtc' | 'fitc':  # more sensible
                    var = kernel(x_test, x_test,
                                 self.kernel_noise, self.kernel_length, diag=True)
                    Lss_inv_Kst = self.Kss.cholesky().solve(Kst)
                    var = var - Lss_inv_Kst.pow(2).sum(dim=0)
                    U_Sigma_T_Kst = self.Sigma.root.T @ Kst
                    var = var + U_Sigma_T_Kst.pow(2).sum(dim=0)
            predictions.append(torch.abs(var))
        return predictions

    def compute_negative_log_marginal_likelihood(self):
        """
        without the size terms as they are not important for optimization
        (TODO: decide whether the include for inspection, or just redefine
               new quantity without it)
        """
        #fsize = self.full_descriptors.shape[1]
        fit_term = -0.5 * self.training_outputs.unsqueeze(0) @ self.Lambda_inv
        fit_term = fit_term @ (self.training_outputs - self.Ksf.T @ self.alpha)
        logdet_Xi_inv = -1 * self.Kss.logdet() - self.Lambda_inv.logdet() - self.Sigma.logdet()
        self.negative_log_marginal_likelihood = 0.5 * logdet_Xi_inv - fit_term  # - fsize * 0.5 * np.log(2 * np.pi))

    def optimize_hyperparameters(self, rtol=1e-2, relax_kernel_length=False):
        """
        Optimize hyperparameters
        """
        def closure():
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            self.__update_Sigma_alpha()
            self.compute_negative_log_marginal_likelihood()
            if self.negative_log_marginal_likelihood.requires_grad:
                self.negative_log_marginal_likelihood.backward()
            return self.negative_log_marginal_likelihood

        self.model_noise.requires_grad_()
        self.kernel_noise.requires_grad_()
        self.kernel_length.requires_grad_(relax_kernel_length)
        counter = 0

        closure()
        d_nlml = np.inf
        prev_nlml = self.negative_log_marginal_likelihood.item()
        while np.abs(d_nlml / prev_nlml) > rtol:
            counter += 1
            self.optimizer.step(closure)
            this_nlml = self.negative_log_marginal_likelihood.item()
            d_nlml = np.abs(this_nlml - prev_nlml)
            print(torch.nn.Softplus()(self.kernel_noise).item() + 1e-7, torch.nn.Softplus()(self.model_noise).item() + 1e-7)
            print(-self.negative_log_marginal_likelihood.item())
            prev_nlml = this_nlml
        self.model_noise.requires_grad_(False)
        self.kernel_noise.requires_grad_(False)
        self.kernel_length.requires_grad_(False)

        self.Lambda_inv = self.Lambda_inv.detach()
        self.Ksf = self.Ksf.detach()
        self.Kss = self.Kss.detach()
        self.Sigma = self.Sigma.detach()
        self.alpha = self.alpha.detach()
        return counter

    def __get_Lambda_inv(self, full_set, update=False):
        """
        Returns the diagonal matrix Λ⁻¹.
        Note: use of full set as input allows for flexible re-use for FITC 
              and during full set updates.
        Args:
            full_set (torch.Tensor): full set of descriptors
            update (bool)          : whether this call is for a full set update
                                     or an initialization/hyperparameter optimization
        """
        size = full_set.shape[1]
        noise = softplus(self.model_noise) + 1e-7  # possibly allow user-flexibility?
        
        match self.variance_mode:
            case 'sor' | 'dtc':
                return ConstantDiagLinearOperator(noise.pow(-2), size)
            case 'fitc':
                Kff = kernel(full_set, full_set,
                             self.kernel_noise, self.kernel_length,
                             diag=True)
                if update:  # do not recompute if not full set update
                    Ksf = kernel(self.sparse_descriptors, full_set,
                                 self.kernel_noise, self.kernel_length)
                else:
                    Ksf = self.Ksf
                aux = self.Kss.cholesky().inverse() @ Ksf
                Lambda_diag = Kff - aux.pow(2).sum(dim=0)
                Lambda_diag = Lambda_diag + noise.pow(2).expand(size)
                return DiagLinearOperator(Lambda_diag).inverse()
