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
      noise = torch.nn.Softplus()(noise) + 1e-7
      return noise * noise * torch.exp(-0.5 * scalar_form / (l * l))


class SparseGaussianProcess():
    """
    TODO: device manipulation
    """
    def __init__(self, descriptor_dim, invert_mode='QR'):
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
        
        #self.optimizer = torch.optim.Rprop([self.model_noise, self.kernel_noise, self.kernel_length], lr=4e-3)

        self.optimizer = LBFGSNew([self.model_noise, self.kernel_noise, self.kernel_length], 
                                   lr=1e-2, history_size=8, max_iter=6)

        self.invert_mode = invert_mode
        # trained model parameters (precomputed during model updates for faster prediction)
        self.Ksf = None
        self.Lambda_inv = None
        self.Kss = None
        self.Sigma = None
        self.alpha = None  # for predicting mean

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
        Kff = kernel(self.full_descriptors, self.full_descriptors,
                     self.kernel_noise, self.kernel_length)
        Kss = kernel(self.sparse_descriptors, self.sparse_descriptors,
                     self.kernel_noise, self.kernel_length)
        self.Ksf = kernel(self.sparse_descriptors, self.full_descriptors,
                          self.kernel_noise, self.kernel_length)
        noise = torch.nn.Softplus()(self.model_noise) + 1e-7

        self.Lambda_inv = ConstantDiagLinearOperator(noise.pow(-2), self.full_descriptors.shape[1])

        Kss = Kss + torch.eye(self.sparse_descriptors.shape[1]) * 1e-8
 
        #start = time.time()
        Lss = torch.linalg.cholesky(Kss)  # M³/3 flops
        #end = time.time()
        #print('Cholesky factorize Kss: {:.5g}s'.format(end - start))
        #print('cond(Lss) = {:.4g}'.format(torch.linalg.cond(Lss).item()))

        self.Kss = CholLinearOperator(Lss, upper=False)
 
        invert_start = time.time()
        match self.invert_mode:
            case 'N':  # use 'Normal Equations' method, i.e. direct Cholesky inverse (very unstable)
                Sigma_inv = Kss + self.Ksf @ self.Lambda_inv @ self.Ksf.T
                Sigma_inv = Sigma_inv + torch.eye(self.sparse_descriptors.shape[1]) * 1  # large jitter!

                #start = time.time()
                L_Sigma_inv = torch.linalg.cholesky(Sigma_inv)
                #end = time.time()
                #print('N method | Cholesky factorize: {:.5g}s'.format(end - start))

                #start = time.time()
                self.U_Sigma = TriangularLinearOperator(L_Sigma_inv, upper=False).inverse().T
                #end = time.time()
                #print('N method | Cholesky inverse: {:.5g}s'.format(end - start))

                #print('N method | cond(L_Σ_inv) = {:.4g}'.format(torch.linalg.cond(L_Sigma_inv).item()))
            case 'V':  # usually stable, sometimes breaks during hyperparameter optimization
                #start = time.time()
                Lss_inv = self.Kss.cholesky().inverse()
                #end = time.time()
                #print('V method | get Lss_inv : {:.5g}s'.format(end - start))

                V = self.Ksf.T @ Lss_inv.T
                Lambda_inv_sqrt_V = self.Lambda_inv.sqrt() @ V
                Gamma = IdentityLinearOperator(V.shape[1]) + Lambda_inv_sqrt_V.T @ Lambda_inv_sqrt_V

                #start = time.time()
                L_Gamma = torch.linalg.cholesky(Gamma.to_dense())
                #end = time.time()
                #print('V method | Cholesky factorize Γ: {:.5g}s'.format(end - start))
                
                #start = time.time()
                U_Sigma = (TriangularLinearOperator(L_Gamma, upper=False).inverse() @ Lss_inv).T
                self.Sigma = RootLinearOperator(U_Sigma)
                #end = time.time()
                #print('V method | Cholesky inverse: {:.5g}s'.format(end - start))

                #print('V method | cond(L_Γ) = {:.4g}'.format(torch.linalg.cond(L_Gamma).item()))
  
            case 'QR':  # very stable
                B = torch.cat([self.Lambda_inv.sqrt() @ self.Ksf.T, Lss.T], dim=0)
                Q, R = stable_qr(B)
                U_Sigma = TriangularLinearOperator(R, upper=True).inverse()
                self.Sigma = RootLinearOperator(U_Sigma)

            case 'L':
                Sigma_inv = Kss + self.Ksf @ self.Lambda_inv @ self.Ksf.T
                self.Sigma = root_inv_decomposition(Sigma_inv, method='symeig')

        invert_end = time.time()
        #print('{} method | Total inversion time: {:.5g}s'.format(self.sigma_mode, invert_end - invert_start))

        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs
        self.alpha = self.Ksf @ self.alpha
        self.alpha = self.Sigma @ self.alpha
 

    def update_full_set(self, x_train, y_train):
        """
        Update full set without inverting entire updated covariance matrix.
        """
        self.full_descriptors = torch.cat((self.full_descriptors, torch.atleast_2d(x_train)), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))
        
        noise = torch.nn.Softplus()(self.model_noise) + 1e-7
        self.Lambda_inv = ConstantDiagLinearOperator(noise.pow(-2), self.full_descriptors.shape[1])

        Ksfprime = kernel(self.sparse_descriptors, torch.atleast_2d(x_train),
                          self.kernel_noise, self.kernel_length)
        self.Ksf = torch.cat([self.Ksf, Ksfprime], dim=1)

        Lambda_prime = ConstantDiagLinearOperator(noise.pow(2), torch.atleast_2d(x_train).shape[1])
        to_invert = Lambda_prime + Ksfprime.T @ self.Sigma @ Ksfprime

        # condition numbers are quite low - stabilizd by the noise term
        #print('Full set update | cond(to_invert) = {:.4g}'.format(torch.linalg.cond(to_invert.to_dense()).item()))

        start = time.time()
        L_to_invert = torch.linalg.cholesky(to_invert.to_dense())
        end = time.time()
        print('Full set update | Cholesky factorization: {:.5g}s'.format(end - start))

        start = time.time()
        L_inv = TriangularLinearOperator(L_to_invert).inverse()
        end = time.time()
        print('Full set update | Cholesky inversion time: {:.5g}s'.format(end - start))

        start = time.time()
        aux = L_inv @ Ksfprime.T @ self.Sigma.root
        Uprime = root_decomposition(IdentityLinearOperator(aux.shape[1]) - aux.T @ aux, method='cholesky').root
        self.Sigma = RootLinearOperator(self.Sigma.root @ Uprime)
        end = time.time()
        print('Full set update | Preserve root: {:.5g}s'.format(end - start))

        self.alpha = self.Lambda_inv @ self.training_outputs
        self.alpha = self.Ksf @ self.alpha
        self.alpha = self.Sigma @ self.alpha

    def update_sparse_set(self, x_sparse):
        """
        
        """
        Kssprime = kernel(self.sparse_descriptors, torch.atleast_2d(x_sparse),
                          self.kernel_noise, self.kernel_length)
        self.sparse_descriptors = torch.cat((self.sparse_descriptors, torch.atleast_2d(x_sparse)), dim=1)
        Ksprimesprime = kernel(torch.atleast_2d(x_sparse), torch.atleast_2d(x_sparse),
                               self.kernel_noise, self.kernel_length)
        Kfsprime = kernel(self.full_descriptors, torch.atleast_2d(x_sparse),
                          self.kernel_noise, self.kernel_length)

        # update Kss
        Lss_inv_Kssp = self.Kss.cholesky().solve(Kssprime)
        Lsp = torch.linalg.cholesky(Ksprimesprime - Lss_inv_Kssp.T @ Lss_inv_Kssp + torch.eye(Ksprimesprime.shape[0]) * 1e-8)
        newLss_upper = torch.cat([self.Kss.cholesky().to_dense(), torch.zeros(Kssprime.shape)], dim=1)
        newLss_lower = torch.cat([Lss_inv_Kssp.T, Lsp.to_dense()], dim=1)
        newLss = torch.cat([newLss_upper, newLss_lower], dim=0)
        self.Kss = CholLinearOperator(newLss, upper=False)

        # update Sigma
        B = Kssprime + self.Ksf @ self.Lambda_inv @ Kfsprime
        Lambda_inv_root_Ksfp = self.Lambda_inv.sqrt() @ Kfsprime
        C = Ksprimesprime + Lambda_inv_root_Ksfp.T @ Lambda_inv_root_Ksfp
        to_invert = C - B.T @ self.Sigma @ B

        # use symeig to preserve shape for assembly
        U_Psi = root_inv_decomposition(to_invert, method='symeig').root

        upper_right = B @ U_Psi
        upper_right = -1 * self.Sigma @ upper_right
        U_Sigma_upper = torch.cat([self.Sigma.root.to_dense(), upper_right], dim=1)
        U_Sigma_lower = torch.cat([torch.zeros(upper_right.T.shape), U_Psi.to_dense()], dim=1)

        U_Sigma = torch.cat([U_Sigma_upper, U_Sigma_lower], dim=0)
        self.Sigma = RootLinearOperator(U_Sigma)

        self.Ksf = torch.cat([self.Ksf, Kfsprime.T], dim=0)
        self.alpha = self.Lambda_inv @ self.training_outputs
        self.alpha = self.Ksf @ self.alpha
        self.alpha = self.Sigma @ self.alpha
 
    def get_predictions(self, x_test, mean_var=[True, True], mode='dtc'):
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
            if mode == 'sor':  # quite nonsensical
                U_Sigma_Kst = self.Sigma.root.T @ Kst
                var = U_Sigma_Kst.pow(2).sum(dim=0)
            elif mode == 'dtc':
                var = kernel(x_test, x_test,
                             self.kernel_noise, self.kernel_length, diag=True)
                Lss_inv_Kst = self.Kss.cholesky().solve(Kst)
                var = var - Lss_inv_Kst.pow(2).sum(dim=0)
                U_Sigma_Kst = self.Sigma.root.T @ Kst
                var = var + U_Sigma_Kst.pow(2).sum(dim=0)
            predictions.append(torch.abs(var))
        return predictions

    def compute_negative_log_marginal_likelihood(self):
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

        self.Kss = self.Kss.detach()
        self.Sigma = self.Sigma.detach()
        self.alpha = self.alpha.detach()
        return counter