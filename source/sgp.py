import numpy as np
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lbfgs'))
from lbfgsnew import LBFGSNew

from utils import jitter, cholesky_update


class SparseGaussianProcess(torch.nn.Module):
    """
    Supports the following SGP approximations
    - subset of regressors (SoR)
    - deterministic training conditional (DTC)
    - fully independent training conditional (FITC)
    - variational free energy (VFE)

    Utilizes efficient model updates for all
    SGP approximations except FITC

    Hyperparameter optimization capabilities
    - noise of data (default)
    - outputscale of kernel (default)
    - sparse set of inducing points (optional)
    - kernel hyperparameters (optional)

    Class attributes:
    - covariance matrices Kss, Ksf (WITHOUT OUTPUTSCALE PREMULTIPLIED)
    """
    def __init__(self,
                 descriptor_dim, kernel,
                 decomp_mode='v', sgp_mode='dtc',
                 init_noise=1e-2, init_outputscale=1.0,
                 noise_range=[1e-4, 2], outputscale_range=[0.1, 10],
                 device='cpu'):
        """
        Args:
            descriptor_dim (int)    : dimensionality of descriptor vector
            kernel (torch.nn.module): SGP kernel object to compute covariance
            decomp_mode (str)       : c, v, qr
            sgp_mode (str)          : sor, dtc, fitc, vfe
            init_noise (float)      : noise for initialization
            init_outputscale (float): outputscale for initialization
            noise_range (list)      : noise hyperparameter range
            outputscale_range (list): outputscale hyperparameter range
            device (torch.device)   : device (default is cpu)
        """
        super().__init__()

        self.device = device
        self.kernel = kernel

        # SGP dataset and sparse inducing points
        self.full_descriptors = torch.empty((descriptor_dim, 0), dtype=torch.float64, device=self.device)
        self.sparse_descriptors = torch.empty((descriptor_dim, 0), dtype=torch.float64, device=self.device)
        self.training_outputs = torch.empty((0,), dtype=torch.float64, device=self.device)

        # basic SGP hyperparameters (noise and outputscale)
        assert ((noise_range[0] > 1e-16) & (noise_range[1] > 1e-16)
                & (init_noise > 1e-16)), 'noise > 1e-16'
        assert ((outputscale_range[0] > 1e-16) & (outputscale_range[1] > 1e-16)
                & (init_outputscale > 1e-16)), 'outputscale > 1e-16'
        assert noise_range[0] < init_noise < noise_range[1]
        assert outputscale_range[0] < init_outputscale < outputscale_range[1]
        self.noise_range = noise_range
        self.outputscale_range = outputscale_range
        noise = torch.tensor([init_noise], dtype=torch.float64, device=self.device)
        self._noise = self.__convert_hyperparameter(noise, noise_range)
        outputscale = torch.tensor([init_outputscale], dtype=torch.float64, device=self.device)
        self._outputscale = self.__convert_hyperparameter(outputscale, outputscale_range)

        # Sigma Cholesky decomposition mode for Sigma and SGP approximations
        assert decomp_mode in ['c', 'v', 'qr'], 'decomp_mode {} not supported \
                                                 only \'c\', \'v\', \'qr\' supported'.format(decomp_mode)
        self.decomp_mode = decomp_mode
        assert sgp_mode in ['sor', 'dtc', 'fitc', 'vfe'], 'only \'sor\', \'dtc\', \'fitc\', \'vfe\' supported'
        self.sgp_mode = sgp_mode

        # unscaled covariance matrices
        self.Kss = None
        self.Ksf = None

        # intermediate GP terms
        self.Lambda_inv = None
        self.Lss = None
        self.L_Sigma = None
        self.alpha = None
        self._nlml = None

    def update_model(self, x_train, y_train, x_sparse=None):
        """
        Update SGP model with training data and/or inducing points. If the model is
        empty (not initialized with training data), it is mandatory to provide
        x_sparse. After initialization, there are three possible use modes:
        1. only update full set            - set x_sparse=None, provide x_train, y_train
        2. only update sparse set          - set x_train=None; provide x_sparse
        3. update both full and sparse set - provide x_train, y_train, x_sparse

        Args:
          x_train (torch.Tensor) : d × m tensor where m is the number of x vectors and d is the
                                   dimensionality of the x descriptor vectors
          y_train (torch.Tensor) : m-dimensional vector of training outputs corresponding to
                                   the training inputs x_train
          x_sparse (torch.Tensor): d × m tensor where m is the number of x vectors and d is the
                                   dimensionality of the x descriptor vectors
        """
        init = (self.full_descriptors.shape[1] == 0)  # no data -> initialize

        if init:
            assert x_sparse is not None, 'model is empty, x_sparse required for initialization'
            self.full_descriptors = torch.cat((self.full_descriptors, x_train), dim=1)
            self.training_outputs = torch.cat((self.training_outputs, y_train))
            self.sparse_descriptors = torch.cat((self.sparse_descriptors, x_sparse), dim=1)
            # compute and keep covariances matrices without outputscale premultiplied
            self.Ksf = self.kernel(self.sparse_descriptors, self.full_descriptors)
            self.Kss = self.kernel(self.sparse_descriptors, self.sparse_descriptors)
            self.__update_intermediates()
        else:
            if self.sgp_mode == 'fitc':  # efficient update doesn't work for FITC
                if x_train is not None:
                    self.full_descriptors = torch.cat((self.full_descriptors, x_train), dim=1)
                    self.training_outputs = torch.cat((self.training_outputs, y_train))
                if x_sparse is not None:
                    self.sparse_descriptors = torch.cat((self.sparse_descriptors, x_sparse), dim=1)
                # compute and keep covariances matrices without outputscale premultiplied
                self.Ksf = self.kernel(self.sparse_descriptors, self.full_descriptors)
                self.Kss = self.kernel(self.sparse_descriptors, self.sparse_descriptors)
                self.__update_intermediates()
            else:
                if x_train is not None:
                    self.__update_full_set(x_train, y_train)
                if x_sparse is not None:
                    self.__update_sparse_set(x_sparse)

    def __update_intermediates(self):
        """
        This function is called to update the intermediate quantities
        Lss, L_Sigma and alpha. This function is only called during
        initialization and for hyperparameter tuning.
        """
        # multiply outputscale to covariance matrices
        outputscale = self.__constrained_hyperparameter('outputscale')
        Ksf = outputscale * self.Ksf
        Kss = outputscale * self.Kss
        # Cholesky decompose Kss with jitter
        self.Lss = torch.linalg.cholesky(jitter(Kss))  # O(M³)
        # get Λ⁻¹
        self.Lambda_inv = self.__get_Lambda_inv(self.full_descriptors)

        match self.decomp_mode:
            case 'c':  # direct Cholesky factorization (least reliable but cheap)
                Sigma = Kss + (Ksf * self.Lambda_inv) @ Ksf.T  # O(M²N)
                self.L_Sigma = torch.linalg.cholesky(jitter(Sigma))  # O(M³)
            case 'v':  # V method (reasonably reliable and moderately expensive)
                V = torch.linalg.solve_triangular(self.Lss, Ksf, upper=False)  # O(M²N)
                V_aux = V * self.Lambda_inv.sqrt()  # O(MN) since Λ⁻¹ is diagonal
                Gamma = jitter(V_aux @ V_aux.T, eps=1)  # O(M²N)
                L_Gamma = torch.linalg.cholesky(Gamma)  # O(M³)
                self.L_Sigma = self.Lss @ L_Gamma  # O(M³)
            case 'qr':  # QR method (most reliable and expensive)
                B = torch.cat([(Ksf * self.Lambda_inv.sqrt()).T, self.Lss.T], dim=0)  # dim (N + M) by M
                _, R = torch.linalg.qr(B)  # O(M²N + M³)
                self.L_Sigma = R.T

        # compute α by doing matrix-vector multiplications first
        self.alpha = torch.mv(Ksf, self.Lambda_inv * self.training_outputs)  # O(MN)
        self.alpha = torch.linalg.solve_triangular(self.L_Sigma, self.alpha.unsqueeze(-1), upper=False)  # O(M²)
        self.alpha = torch.linalg.solve_triangular(self.L_Sigma.T, self.alpha, upper=True)[:, 0]  # O(M²)

    def __update_full_set(self, x_train, y_train):
        """
        Update model with N' new full set data points (input and output).
        """
        self.full_descriptors = torch.cat((self.full_descriptors, torch.atleast_2d(x_train)), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))

        # update covariance matrices
        outputscale = self.__constrained_hyperparameter('outputscale')
        Ksfprime = self.kernel(self.sparse_descriptors, x_train)
        self.Ksf = torch.cat([self.Ksf, Ksfprime], dim=1)
        Ksfprime = outputscale * Ksfprime
        Ksf = outputscale * self.Ksf

        # Update Σ (U_Σ)
        Lambda_inv_prime = self.__get_Lambda_inv(x_train, update=True)
        aux = Ksfprime * Lambda_inv_prime.sqrt()
        self.L_Sigma = cholesky_update(self.L_Sigma, aux)  # use efficient Cholesky update
        self.Lambda_inv = torch.cat([self.Lambda_inv, Lambda_inv_prime])

        # compute α by doing matrix-vector multiplications first
        self.alpha = torch.mv(Ksf, self.Lambda_inv * self.training_outputs)  # O(MN)
        self.alpha = torch.linalg.solve_triangular(self.L_Sigma, self.alpha.unsqueeze(-1), upper=False)  # O(M²)
        self.alpha = torch.linalg.solve_triangular(self.L_Sigma.T, self.alpha, upper=True)[:, 0]  # O(M²)

    def __update_sparse_set(self, x_sparse):
        """
        Update model with M' new sparse set points.
        """
        # update covariance matrices
        outputscale = self.__constrained_hyperparameter('outputscale')
        # Kss
        Kssprime = self.kernel(self.sparse_descriptors, x_sparse)
        Ksprimesprime = self.kernel(torch.atleast_2d(x_sparse), torch.atleast_2d(x_sparse))
        Kss_upper = torch.cat([self.Kss, Kssprime], dim=1)
        Kss_lower = torch.cat([Kssprime.T, Ksprimesprime], dim=1)
        self.Kss = torch.cat([Kss_upper, Kss_lower], dim=0)
        # Ksf
        Kfsprime = self.kernel(self.full_descriptors, x_sparse)  # without outputscale
        Ksf_prev = outputscale * self.Ksf  # required later
        self.Ksf = torch.cat([self.Ksf, Kfsprime.T], dim=0)  # without outputscale

        self.sparse_descriptors = torch.cat((self.sparse_descriptors, x_sparse), dim=1)

        # multiply in outputscale
        Kssprime = outputscale * Kssprime
        Ksprimesprime = outputscale * Ksprimesprime
        Kfsprime = outputscale * Kfsprime

        # update Lss with block trick
        Lss_inv_Kssp = torch.linalg.solve_triangular(self.Lss, Kssprime, upper=False)  # O(M²M')
        Lsp = torch.linalg.cholesky(jitter(Ksprimesprime - Lss_inv_Kssp.T @ Lss_inv_Kssp))  # O(M'³)
        zero_block = torch.zeros(Kssprime.shape, dtype=Kssprime.dtype, device=self.device)
        newLss_upper = torch.cat([self.Lss, zero_block], dim=1)
        newLss_lower = torch.cat([Lss_inv_Kssp.T, Lsp], dim=1)
        self.Lss = torch.cat([newLss_upper, newLss_lower], dim=0)

        # Update Σ (U_Σ) with block trick
        B = Kssprime + (Ksf_prev * self.Lambda_inv) @ Kfsprime  # O(NMM')
        Kspf_Lamda_inv_root = Kfsprime.T * self.Lambda_inv.sqrt()  # O(NM') since Λ⁻¹ is diagonal
        C = Ksprimesprime + Kspf_Lamda_inv_root @ Kspf_Lamda_inv_root.T  # O(N²M')
        aux = torch.linalg.solve_triangular(self.L_Sigma, B, upper=False)  # O(M²M')
        schur = C - aux.T @ aux  # O(MM'²)
        # get nearest psd schur (https://doi.org/10.1016/0024-3795(88)90223-6)
        schur = 0.5 * (schur + schur.T)
        L, Q = torch.linalg.eigh(schur)
        schur_root = Q * torch.clamp(L, min=0).sqrt()
        L_Psi = torch.linalg.cholesky(jitter(schur_root @ schur_root.T))
        # assemble new L_Sigma
        zero_block = torch.zeros((self.L_Sigma.shape[0], L_Psi.shape[1]), dtype=L_Psi.dtype, device=self.device)
        L_upper = torch.cat([self.L_Sigma, zero_block], dim=1)
        L_lower = torch.cat([aux.T, L_Psi], dim=1)
        self.L_Sigma = torch.cat([L_upper, L_lower], dim=0)

        self.alpha = torch.mv(outputscale * self.Ksf, self.Lambda_inv * self.training_outputs)  # O(MN)
        self.alpha = torch.linalg.solve_triangular(self.L_Sigma, self.alpha.unsqueeze(-1), upper=False)  # O(M²)
        self.alpha = torch.linalg.solve_triangular(self.L_Sigma.T, self.alpha, upper=True)[:, 0]  # O(M²)

    def forward(self, x_test, mean_var=[True, True], include_noise=False):
        """
        Get predictions of GP model with a set of testing vectors.

        Args:
          x_test (torch.Tensor): d × p tensor where p is the number of x vectors and d is the
                                 dimensionality of the x descriptor vectors
          mean_var (bool list) : whether the function returns the mean or variance or both
          include_noise (bool) : whether the noise is included in the variance
        """
        predictions = []
        # handle possibility of inference with uninitialized model
        if (self.full_descriptors.shape[1] == 0):
            if mean_var[0]:
                mean = torch.zeros(x_test.shape[1], dtype=x_test.dtype, device=self.device)
                predictions.append(mean)
            if mean_var[1]:
                noise = self.__constrained_hyperparameter('noise')
                var = noise.expand(x_test.shape[1])
                predictions.append(var)
        else:
            # compute covariance matrix
            outputscale = self.__constrained_hyperparameter('outputscale')
            Kst = outputscale * self.kernel(self.sparse_descriptors, x_test)
            if mean_var[0]:
                mean = torch.mv(Kst.T, self.alpha)
                predictions.append(mean)
            if mean_var[1]:
                match self.sgp_mode:
                    case 'sor':  # quite nonsensical
                        LSigma_inv_Kst = torch.linalg.solve_triangular(self.L_Sigma, Kst, upper=False)
                        var = LSigma_inv_Kst.pow(2).sum(dim=0)
                    case 'dtc' | 'fitc' | 'vfe':  # more sensible
                        var = outputscale * self.kernel(x_test, x_test, diag=True)
                        Lss_inv_Kst = torch.linalg.solve_triangular(self.Lss, Kst, upper=False)
                        var = var - Lss_inv_Kst.pow(2).sum(dim=0)
                        LSigma_inv_Kst = torch.linalg.solve_triangular(self.L_Sigma, Kst, upper=False)
                        var = var + LSigma_inv_Kst.pow(2).sum(dim=0)
                if include_noise:
                    var = var + self.__constrained_hyperparameter('noise').pow(2)
                predictions.append(torch.abs(var))
        return predictions

    def __compute_negative_log_marginal_likelihood(self):
        """
        Internal function to compute the negative log marginal likelihood self._nlml,
        a quantity to be minimized during hyperparameter optimization.
        Note: self._nlml is not updated unless this function or optimize_hyperparameters is called.
        """
        outputscale = self.__constrained_hyperparameter('outputscale')
        aux = self.training_outputs - outputscale * torch.mv(self.Ksf.T, self.alpha)
        fit = 0.5 * torch.dot(self.training_outputs, self.Lambda_inv * aux)
        penalty = -1 * self.Lss.logdet() - 0.5 * torch.sum(torch.log(self.Lambda_inv))
        penalty = penalty + torch.log(torch.abs(torch.sum(self.L_Sigma.diag())))
        size = self.full_descriptors.shape[1] * 0.5 * np.log(2 * np.pi)
        if self.sgp_mode == 'vfe':
            noise = self.__constrained_hyperparameter('noise')
            outputscale = self.__constrained_hyperparameter('outputscale')
            Kff = outputscale * self.kernel(self.full_descriptors, self.full_descriptors, diag=True)
            Ksf = outputscale * self.Ksf
            aux = torch.linalg.solve_triangular(self.Lss, Ksf, upper=False)
            trace = torch.sum(Kff - aux.pow(2).sum(dim=0)) / noise.pow(2).mul(2)
            self._nlml = penalty + fit + trace + size
        else:
            self._nlml = penalty + fit + size

    @property
    def log_marginal_likelihood(self):
        self.__compute_negative_log_marginal_likelihood()
        return -1 * self._nlml.item()

    def optimize_hyperparameters(self, rtol=1e-2, relax_inducing_points=False, relax_kernel_params=False):
        """
        Optimize SGP (and kernel) hyperparameters. This function will always optimize
        the noise and kernel outputscale by default. It can additionally optimize the
        inducing points and other kernel parameters.

        Args:
            rtol (float)                : relative tolerance for terminating optimization
            relax_inducing_points (bool): whether to relax the inducing points or not
            relax_kernel_params (bool)  : whether to relax kernel hyperparameters
        """
        # assemble hyperparameters to be optimized
        params = [self._noise, self._outputscale]
        if relax_kernel_params:
            params += self.kernel.kernel_hyperparameters
        if relax_inducing_points:
            params.append(self.sparse_descriptors)
        # initialize LBFGS optimizer
        self.optimizer = LBFGSNew(params, lr=2e-3, history_size=8, max_iter=5)

        def closure():
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            if relax_inducing_points or relax_kernel_params:
                self.Ksf = self.kernel(self.sparse_descriptors, self.full_descriptors)
                self.Kss = self.kernel(self.sparse_descriptors, self.sparse_descriptors)
            self.__update_intermediates()
            self.__compute_negative_log_marginal_likelihood()
            if self._nlml.requires_grad:
                self._nlml.backward()
            return self._nlml

        for param in params:
            param.requires_grad_()
        counter = 0
        closure()
        with open('hypopt.dat', 'w') as f:
            f.write('{:^15} {:^15} {:^15}\n'
                    .format('NLML', 'Outputscale', 'Noise'))
            f.write('{:^15.8g} {:^15.8g} {:^15.8g}\n'
                    .format(-self._nlml.item(), self.outputscale, self.noise))

        d_nlml = np.inf
        prev_nlml = self._nlml.item()
        while np.abs(d_nlml / prev_nlml) > rtol:
            counter += 1
            self.optimizer.step(closure)
            this_nlml = self._nlml.item()
            d_nlml = np.abs(this_nlml - prev_nlml)
            with open('hypopt.dat', 'a') as f:
                f.write('{:^15.8g} {:^15.8g} {:^15.8g}\n'
                        .format(-self._nlml.item(), self.outputscale, self.noise))
            prev_nlml = this_nlml

        for param in params:
            param.requires_grad_(False)

        # detach intermediate variables
        self.Lambda_inv = self.Lambda_inv.detach()
        self.Lss = self.Lss.detach()
        self.L_Sigma = self.L_Sigma.detach()
        self.alpha = self.alpha.detach()
        self._nlml = self._nlml.detach()
        if relax_inducing_points or relax_kernel_params:
            self.Ksf = self.Ksf.detach()
            self.Kss = self.Kss.detach()
        return counter

    def __get_Lambda_inv(self, full_set, update=False):
        """
        Internal function that returns the diagonal matrix Λ⁻¹.
        Note: use of full set as input allows for flexible re-use for FITC
              and during full set updates.
        Args:
            full_set (torch.Tensor): full set of descriptors
            update (bool)          : whether this call is for a full set update
                                     or an initialization/hyperparameter optimization
        """
        size = full_set.shape[1]
        noise = self.__constrained_hyperparameter('noise')
        match self.sgp_mode:
            case 'sor' | 'dtc' | 'vfe':
                return noise.pow(-2).expand(size)
            case 'fitc':
                outputscale = self.__constrained_hyperparameter('outputscale')
                Kff = outputscale * self.kernel(full_set, full_set, diag=True)
                if update:  # do not recompute if not full set update
                    Ksf = outputscale * self.kernel(self.sparse_descriptors, full_set)
                else:
                    Ksf = outputscale * self.Ksf
                aux = torch.linalg.solve_triangular(self.Lss, Ksf, upper=False)
                Lambda_diag = Kff - aux.pow(2).sum(dim=0)
                Lambda_diag = Lambda_diag + noise.pow(2).expand(size)
                return Lambda_diag.pow(-1)

    def __constrained_hyperparameter(self, hyperparameter):
        match hyperparameter:
            case 'outputscale':
                return self.outputscale_range[0] + \
                       (self.outputscale_range[1] - self.outputscale_range[0]) * torch.sigmoid(self._outputscale)
            case 'noise':
                return self.noise_range[0] + (self.noise_range[1] - self.noise_range[0]) * torch.sigmoid(self._noise)

    def __convert_hyperparameter(self, hparam, param_range):
        # used for initializing hyperparameters
        return - torch.log(((param_range[1] - param_range[0]) / (hparam - param_range[0])) - 1)

    @property
    def noise(self):
        return self.__constrained_hyperparameter('noise').item()

    @property
    def outputscale(self):
        return self.__constrained_hyperparameter('outputscale').item()
