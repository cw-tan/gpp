import numpy as np
import torch
from linear_operator.operators import IdentityLinearOperator, ConstantDiagLinearOperator, \
                                      DiagLinearOperator, TriangularLinearOperator, \
                                      RootLinearOperator
from linear_operator import settings, root_decomposition, root_inv_decomposition
from linear_operator.utils import stable_qr

import time
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lbfgs'))
from lbfgsnew import LBFGSNew


# Extensions:
# 1. derivatives - need to redesign how SGP and Kernel classes interact?
# 2. add more kernels?

# Low priority aesthetic/user-friendliness TODOs:
# 1. proper torch.nn parameter monitoring (may make code cleaner, but more complex)
# 2. clean up docstrings
# 3. verbosity for operations? timing etc? may clutter code though

class SparseGaussianProcess(torch.nn.Module):
    """
    Supports the following SGP approximations
    - subset of regressors (SoR)
    - deterministic training conditional (DTC)
    - fully independent training conditional (FITC)
    - variational free energy (VFE)

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
                 invert_mode='v', sgp_mode='dtc',
                 init_noise=1e-2, init_outputscale=1.0,
                 noise_range=[1e-4, 2], outputscale_range=[0.1, 10]):
        """
        Args:
            descriptor_dim (int)    : dimensionality of descriptor vector
            invert_mode (str)       : c, v, qr
            sgp_mode (str)          : sor, dtc, fitc, vfe
            init_noise (float)      : noise for initialization
            init_outputscale (float): outputscale for initialization
            noise_range (list)      : noise hyperparameter range
            outputscale_range (list): outputscale hyperparameter range
        """
        super().__init__()
        # model data (maybe data should be kernel attributes)
        self.full_descriptors = torch.empty((descriptor_dim, 0))
        self.sparse_descriptors = torch.empty((descriptor_dim, 0))
        self.training_outputs = torch.empty((0,))

        self.kernel = kernel

        # basic SGP hyperparameters (noise and outputscale)
        assert ((noise_range[0] > 1e-16) & (noise_range[1] > 1e-16)
                & (init_noise > 1e-16)), 'noise > 1e-16'
        assert ((outputscale_range[0] > 1e-16) & (outputscale_range[1] > 1e-16)
                & (init_outputscale > 1e-16)), 'outputscale > 1e-16'
        assert noise_range[0] < init_noise < noise_range[1]
        assert outputscale_range[0] < init_outputscale < outputscale_range[1]
        self.noise_range = noise_range
        self.outputscale_range = outputscale_range
        self._noise = self.__convert_hyperparameter(torch.tensor([init_noise], dtype=torch.float64), noise_range)
        self._outputscale = self.__convert_hyperparameter(torch.tensor([init_outputscale], dtype=torch.float64),
                                                          outputscale_range)

        # inversion mode for Sigma and SGP approximations
        assert invert_mode in ['c', 'v', 'qr'], 'only \'c\', \'v\', \'qr\' supported'
        self.invert_mode = invert_mode
        assert sgp_mode in ['sor', 'dtc', 'fitc', 'vfe'], 'only \'sor\', \'dtc\', \'fitc\', \'vfe\' supported'
        self.sgp_mode = sgp_mode

        # covariance matrices
        self.Kss = None
        self.Ksf = None

        # intermediate GP terms (recalculated during hyperparameter optimization)
        self.Lambda_inv = None
        self.Lss = None
        self.Sigma = None
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
            self.__update_Lss_Sigma_alpha()
        else:  # updating sparse set first is more efficient as U_Sigma gets low rank update
            if x_sparse is not None:
                self.__update_sparse_set(x_sparse)
            if x_train is not None:
                self.__update_full_set(x_train, y_train)

    def __update_Lss_Sigma_alpha(self):
        """
        This function is called for initialization and for hyperparameter tuning.
        """
        # multiply outputscale to covariance matrices
        outputscale = self.__constrained_hyperparameter('outputscale')
        Ksf = outputscale * self.Ksf
        Kss = outputscale * self.Kss
        # Cholesky decompose Kss with jitter
        self.Lss = torch.linalg.cholesky(Kss + torch.eye(Kss.shape[0]) * 1e-8)
        # get Λ⁻¹
        self.Lambda_inv = self.__get_Lambda_inv(self.full_descriptors)

        # invert_start = time.time()
        match self.invert_mode:
            case 'c':  # direct Cholesky inverse (most unstable)
                Sigma_inv = Kss + Ksf @ self.Lambda_inv @ Ksf.T  # O(M²N)
                Sigma_inv = Sigma_inv + torch.eye(self.sparse_descriptors.shape[1])  # large jitter!
                L_Sigma_inv = torch.linalg.cholesky(Sigma_inv)  # O(M³)
                U_Sigma = TriangularLinearOperator(L_Sigma_inv, upper=False).inverse().T  # O(M³)
                # print('N method | cond(L_Σ_inv) = {:.4g}'.format(torch.linalg.cond(L_Sigma_inv).item()))
            case 'v':  # V method (usually stable)
                V = torch.linalg.solve_triangular(self.Lss, Ksf, upper=False)  # O(M²N)
                V_Lambda_inv_sqrt = V @ self.Lambda_inv.sqrt()  # O(MN) since Λ⁻¹ is diagonal
                Gamma = torch.eye(V.shape[0], dtype=torch.float64) + V_Lambda_inv_sqrt @ V_Lambda_inv_sqrt.T  # O(M²N)
                L_Gamma = torch.linalg.cholesky(Gamma)  # O(M³)
                A = L_Gamma.T @ self.Lss.T  # O(M³) to form A and solve A U_Sigma =I
                U_Sigma = torch.linalg.solve_triangular(A, torch.eye(A.shape[0], dtype=torch.float64), upper=True)
                # print('V method | cond(L_Γ) = {:.4g}'.format(torch.linalg.cond(L_Gamma).item()))
            case 'qr':  # QR method (most stable)
                B = torch.cat([self.Lambda_inv.sqrt() @ Ksf.T, self.Lss.T], dim=0)  # (N + M) by M
                _, R = stable_qr(B)  # O(M²N + M³)
                U_Sigma = torch.linalg.solve_triangular(R, torch.eye(R.shape[0], dtype=torch.float64),
                                                        upper=True)  # O(M³)
                # print('QR method | cond(R) = {:.4g}'.format(torch.linalg.cond(R).item()))
        # invert_end = time.time()
        # print('{} method | Total inversion time: {:.5g}s'.format(self.invert_mode, invert_end - invert_start))

        self.Sigma = RootLinearOperator(U_Sigma)
        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N) since Λ⁻¹ is diagonal
        self.alpha = Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

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
        Lambda_prime = self.__get_Lambda_inv(x_train, update=True).inverse()
        Kfps_U = Ksfprime.T @ self.Sigma.root  # O(M²N')
        to_invert = Lambda_prime + Kfps_U @ Kfps_U.T  # O(MN'²)
        L_to_invert = torch.linalg.cholesky(to_invert.to_dense())  # O(N'³)

        C = TriangularLinearOperator(L_to_invert).solve(Ksfprime.T @ self.Sigma.root)
        I_minus_CCT = IdentityLinearOperator(C.shape[1]) - C.T @ C  # O(M²N')
        Uprime = root_decomposition(I_minus_CCT, method='cholesky').root  # O(M³)

        self.Sigma = RootLinearOperator(self.Sigma.root @ Uprime)  # O(M³) matmul
        self.Lambda_inv = DiagLinearOperator(torch.cat([self.Lambda_inv.diagonal(),
                                                        Lambda_prime.inverse().diagonal()]))
        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N) since Λ⁻¹ is diagonal
        self.alpha = Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

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
        Lsp = torch.linalg.cholesky(Ksprimesprime - Lss_inv_Kssp.T @ Lss_inv_Kssp
                                    + torch.eye(Ksprimesprime.shape[0]) * 1e-8)  # O(M'³)
        newLss_upper = torch.cat([self.Lss, torch.zeros(Kssprime.shape)], dim=1)
        newLss_lower = torch.cat([Lss_inv_Kssp.T, Lsp], dim=1)
        self.Lss = torch.cat([newLss_upper, newLss_lower], dim=0)
        # print('Lss: {:.5g}'.format(torch.linalg.cond(self.Lss).item()))

        # Update Σ (U_Σ) with block trick
        Lambda_inv_Ksfp = self.Lambda_inv @ Kfsprime  # O(NM') since Λ⁻¹ is diagonal
        B = Kssprime + Ksf_prev @ Lambda_inv_Ksfp  # O(NMM')
        Lambda_inv_root_Kfsp = self.Lambda_inv.sqrt() @ Kfsprime  # O(NM') since Λ⁻¹ is diagonal
        C = Ksprimesprime + Lambda_inv_root_Kfsp.T @ Lambda_inv_root_Kfsp  # O(N²M')
        aux = B.T @ self.Sigma.root  # O(M²M')
        to_invert = C - aux @ aux.T  # O(MM'²)

        # get root decomposition of Psi
        U_Psi = root_inv_decomposition(to_invert, method='svd').root.to_dense()  # O(M'³)
        # RQ decomposition to make U_Psi upper triangular
        P1 = torch.fliplr(torch.eye(U_Psi.shape[0], dtype=torch.float64))
        P2 = torch.fliplr(torch.eye(U_Psi.shape[1], dtype=torch.float64))
        _, R = torch.linalg.qr((P1 @ U_Psi).T, mode='complete')
        R = P1 @ R.T @ P2
        U_Psi = R

        upper_right = B @ U_Psi  # O(MM'²)
        upper_right = -1 * self.Sigma @ upper_right  # O(M²M')
        U_Sigma_upper = torch.cat([self.Sigma.root.to_dense(), upper_right], dim=1)
        U_Sigma_lower = torch.cat([torch.zeros((U_Psi.shape[0], self.Sigma.root.shape[1])), U_Psi.to_dense()], dim=1)
        U_Sigma = torch.cat([U_Sigma_upper, U_Sigma_lower], dim=0)
        self.Sigma = RootLinearOperator(U_Sigma)

        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N) since Λ⁻¹ is diagonal
        self.alpha = outputscale * self.Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

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
                mean = torch.zeros(x_test.shape[1], dtype=torch.float64)
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
                mean = Kst.T @ self.alpha
                predictions.append(mean)
            if mean_var[1]:
                match self.sgp_mode:
                    case 'sor':  # quite nonsensical
                        U_Sigma_Kst = self.Sigma.root.T @ Kst
                        var = U_Sigma_Kst.pow(2).sum(dim=0)
                    case 'dtc' | 'fitc' | 'vfe':  # more sensible
                        var = outputscale * self.kernel(x_test, x_test, diag=True)
                        Lss_inv_Kst = torch.linalg.solve_triangular(self.Lss, Kst, upper=False)
                        var = var - Lss_inv_Kst.pow(2).sum(dim=0)
                        U_Sigma_T_Kst = self.Sigma.root.T @ Kst
                        var = var + U_Sigma_T_Kst.pow(2).sum(dim=0)
                if include_noise:
                    var = var + self.__constrained_hyperparameter('noise').pow(2)
                predictions.append(torch.abs(var))
        return predictions

    def __compute_negative_log_marginal_likelihood(self):
        """
        Internal function to compute the negative log marginal likelihood self._nlml,
        a quantity to be minimized during hyperparameter optimization.
        Note:
        1. self._nlml is not updated unless this function or optimize_hyperparameters is called.
        2. the size term is neglected as they are not important for optimization (for now)
        TODO: decide whether the include size term
        """
        outputscale = self.__constrained_hyperparameter('outputscale')
        fit = 0.5 * self.training_outputs.unsqueeze(0) @ self.Lambda_inv
        fit = fit @ (self.training_outputs - outputscale * self.Ksf.T @ self.alpha)
        penalty = -1 * self.Lss.logdet() - 0.5 * self.Lambda_inv.logdet() - 0.5 * self.Sigma.logdet()
        # size = self.full_descriptors.shape[1] * 0.5 * np.log(2 * np.pi)
        if self.sgp_mode == 'vfe':
            noise = self.__constrained_hyperparameter('noise')
            outputscale = self.__constrained_hyperparameter('outputscale')
            Kff = outputscale * self.kernel(self.full_descriptors, self.full_descriptors, diag=True)
            Ksf = outputscale * self.Ksf
            aux = torch.linalg.solve_triangular(self.Lss, Ksf, upper=False)
            trace = torch.sum(Kff - aux.pow(2).sum(dim=0)) / (noise.pow(2).mul(2))
            self._nlml = penalty + fit + trace  # +size
        else:
            self._nlml = penalty + fit  # + size

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

        self.optimizer = LBFGSNew(params, lr=1e-2, history_size=8, max_iter=5)

        def closure():
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            if relax_inducing_points or relax_kernel_params:
                self.Ksf = self.kernel(self.sparse_descriptors, self.full_descriptors)
                self.Kss = self.kernel(self.sparse_descriptors, self.sparse_descriptors)
            self.__update_Lss_Sigma_alpha()
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
        self.Sigma = self.Sigma.detach()
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
                return ConstantDiagLinearOperator(noise.pow(-2), size)
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
                return DiagLinearOperator(Lambda_diag).inverse()

    def __constrained_hyperparameter(self, hyperparameter):
        match hyperparameter:
            case 'outputscale':
                return self.outputscale_range[0] + self.outputscale_range[1] * torch.sigmoid(self._outputscale)
            case 'noise':
                return self.noise_range[0] + self.noise_range[1] * torch.sigmoid(self._noise)

    def __convert_hyperparameter(self, hparam, param_range):
        # used for initializing hyperparameters
        return - torch.log((param_range[1] / (hparam - param_range[0])) - 1)

    @property
    def noise(self):
        return self.__constrained_hyperparameter('noise').item()

    @property
    def outputscale(self):
        return self.__constrained_hyperparameter('outputscale').item()
