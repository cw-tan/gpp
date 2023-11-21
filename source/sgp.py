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

# Notes:
# 1. linear operator Cholesky decomposition seems slower than native torch on cpu

softplus = torch.nn.Softplus()

# TODOs:
# 1. add test for equivalence of update order
# 2. make kernel a model child and have it carry its own hyperparameters
# 3. centralize model update interface (instead of separate sparse and full set updates)
# 4. logging for hyperparameter optimization
# 5. not important - set correct behavior for uninitialized model


def kernel(x1, x2, lengthscale, diag=False):
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
    return torch.exp(-0.5 * scalar_form / lengthscale.pow(2))


class SparseGaussianProcess():
    """
    class attributes (what you'd get if you do object.attribute):

        covariance matrices Kss, Ksf (WITHOUT OUTPUTSCALE PREMULTIPLIED)

    """
    def __init__(self,
                 descriptor_dim,
                 invert_mode='v', sgp_mode='dtc',
                 init_noise=1e-2, init_outputscale=1.0,
                 noise_range=[1e-4, 2], outputscale_range=[0.1, 10]):
        """
        Args:
            descriptor_dim (int)    : dimensionality of descriptor vector
            invert_mode (str)       : v, qr
            sgp_mode (str)     : sor, dtc, fitc
            noise_range (list)      : noise hyperparameter range
            outputscale_range (list): outputscale hyperparameter range
        """
        # model data
        # self.descriptor_dim = descriptor_dim
        self.full_descriptors = torch.empty((descriptor_dim, 0))
        self.sparse_descriptors = torch.empty((descriptor_dim, 0))
        self.training_outputs = torch.empty((0,))

        # hyperparameters
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

        self.kernel_length = torch.tensor([1], dtype=torch.float64)

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
        self.full_descriptors = torch.cat((self.full_descriptors, x_train), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))
        self.sparse_descriptors = torch.cat((self.sparse_descriptors, x_sparse), dim=1)

        # compute and keep covariances matrices without outputscale premultiplied
        self.Ksf = kernel(self.sparse_descriptors, self.full_descriptors, self.kernel_length)
        self.Kss = kernel(self.sparse_descriptors, self.sparse_descriptors, self.kernel_length)

        self.__update_Lss_Sigma_alpha()
        # TODO: use this function to unify initialization, full set and sparse set updates

    def get_duplicate_ids(self, x1, x2, tol=1e-8):
        """
        Get indices of duplicate points based on a kernel function with output range [0,1]
        within given tolerance. The duplicate indices correspond to that of x2 (i.e. we
        keep x1 as is and remove the duplicate indices from x2).

        Args:
            x1, x2 (torch.Tensor): d × m tensors where d is the dimensionality and m is the number
                                   of descriptor vectors
            tol (float)          : how close the entries of K(x1, x2) are to 1
        """
        if x1.shape[1] > x2.shape[1]:
            x_larger, x_smaller = x1, x2
        else:
            x_larger, x_smaller = x2, x1
        triu_ids = torch.triu_indices(x_smaller.shape[1], x_larger.shape[1], offset=int(torch.equal(x1, x2)))
        K = kernel(x_smaller, x_larger, self.kernel_length)
        duplicate_triu_ids = torch.nonzero((1 - K[triu_ids[0], triu_ids[1]]) < tol, as_tuple=True)[0]
        if x1.shape[1] > x2.shape[1]:
            duplicate_ids = triu_ids[0][duplicate_triu_ids]
        else:
            duplicate_ids = triu_ids[1][duplicate_triu_ids]
        return duplicate_ids

    def remove_duplicates(self, x1, x2, tol=1e-8):
        """
        Returns x2 with duplicates (based on comparisons to x1) removed
        within some tolerance.
        """
        ids_to_remove = self.get_duplicate_ids(x1, x2, tol)
        mask = torch.ones(x2.shape[1], dtype=torch.bool)
        mask[ids_to_remove] = False
        all_ids = torch.arange(x2.shape[1])
        ids_to_keep = all_ids[mask]
        return x2[:, ids_to_keep]

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

        invert_start = time.time()
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
        invert_end = time.time()
        print('{} method | Total inversion time: {:.5g}s'.format(self.invert_mode, invert_end - invert_start))

        self.Sigma = RootLinearOperator(U_Sigma)
        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N) since Λ⁻¹ is diagonal
        self.alpha = Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

    def update_full_set(self, x_train, y_train):
        """
        Update model with N' full points.
        """
        self.full_descriptors = torch.cat((self.full_descriptors, torch.atleast_2d(x_train)), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))

        # update covariance matrices
        outputscale = self.__constrained_hyperparameter('outputscale')
        Ksfprime = kernel(self.sparse_descriptors, torch.atleast_2d(x_train), self.kernel_length)
        self.Ksf = torch.cat([self.Ksf, Ksfprime], dim=1)
        Ksfprime = outputscale * Ksfprime
        Ksf = outputscale * self.Ksf

        # Update Σ (U_Σ)
        Lambda_prime = self.__get_Lambda_inv(torch.atleast_2d(x_train), update=True).inverse()
        Kfps_U = Ksfprime.T @ self.Sigma.root  # O(M²N')
        to_invert = Lambda_prime + Kfps_U @ Kfps_U.T  # O(MN'²)
        L_to_invert = torch.linalg.cholesky(to_invert.to_dense())  # O(N'³)

        C = TriangularLinearOperator(L_to_invert).solve(Ksfprime.T @ self.Sigma.root)
        I_minus_CCT = IdentityLinearOperator(C.shape[1]) - C.T @ C  # O(M²N')
        # TODO: can we use Lanczos here more stably?
        Uprime = root_decomposition(I_minus_CCT, method='cholesky').root  # O(M³)
        self.Sigma = RootLinearOperator(self.Sigma.root @ Uprime)  # O(M³) matmul
        self.Lambda_inv = DiagLinearOperator(torch.cat([self.Lambda_inv.diagonal(),
                                                        Lambda_prime.inverse().diagonal()]))
        # compute α by doing matrix-vector multiplications first
        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N) since Λ⁻¹ is diagonal
        self.alpha = Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

    def update_sparse_set(self, x_sparse):
        """
        Update model with M' new sparse points
        """
        # update covariance matrices
        outputscale = self.__constrained_hyperparameter('outputscale')
        # Kss
        Kssprime = kernel(self.sparse_descriptors, torch.atleast_2d(x_sparse), self.kernel_length)
        Ksprimesprime = kernel(torch.atleast_2d(x_sparse), torch.atleast_2d(x_sparse), self.kernel_length)
        Kss_upper = torch.cat([self.Kss, Kssprime], dim=1)
        Kss_lower = torch.cat([Kssprime.T, Ksprimesprime], dim=1)
        self.Kss = torch.cat([Kss_upper, Kss_lower], dim=0)
        # Ksf
        Kfsprime = kernel(self.full_descriptors, torch.atleast_2d(x_sparse), self.kernel_length)  # without outputscale
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

        # lanczos is stable because of low-rank approximation
        U_Psi = root_inv_decomposition(to_invert, method='lanczos').root  # O(M'³)

        upper_right = B @ U_Psi  # O(MM'²)
        upper_right = -1 * self.Sigma @ upper_right  # O(M²M')
        U_Sigma_upper = torch.cat([self.Sigma.root.to_dense(), upper_right], dim=1)
        U_Sigma_lower = torch.cat([torch.zeros((U_Psi.shape[0], self.Sigma.root.shape[1])), U_Psi.to_dense()], dim=1)
        U_Sigma = torch.cat([U_Sigma_upper, U_Sigma_lower], dim=0)
        self.Sigma = RootLinearOperator(U_Sigma)

        self.alpha = self.Lambda_inv @ self.training_outputs  # O(N) since Λ⁻¹ is diagonal
        self.alpha = outputscale * self.Ksf @ self.alpha  # O(MN)
        self.alpha = self.Sigma @ self.alpha  # O(M²)

    def get_predictions(self, x_test, mean_var=[True, True], include_noise=False):
        """
        Get predictions of GP model with a set of testing vectors.

        Args:
          x_test (torch.Tensor): d × p tensor where p is the number of x vectors and d is the
                                  dimensionality of the x descriptor vectors
        """
        # compute covariance matrix
        outputscale = self.__constrained_hyperparameter('outputscale')
        Kst = outputscale * kernel(self.sparse_descriptors, x_test, self.kernel_length)
        predictions = []
        if mean_var[0]:
            mean = Kst.T @ self.alpha
            predictions.append(mean)
        if mean_var[1]:
            match self.sgp_mode:
                case 'sor':  # quite nonsensical
                    U_Sigma_Kst = self.Sigma.root.T @ Kst
                    var = U_Sigma_Kst.pow(2).sum(dim=0)
                case 'dtc' | 'fitc' | 'vfe':  # more sensible
                    var = outputscale * kernel(x_test, x_test, self.kernel_length, diag=True)
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
            Kff = outputscale * kernel(self.full_descriptors, self.full_descriptors,
                                       self.kernel_length, diag=True)
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
        params = [self._noise, self._outputscale]
        if relax_kernel_params:
            params.append(self.kernel_length)
        if relax_inducing_points:
            params.append(self.sparse_descriptors)

        self.optimizer = LBFGSNew(params, lr=1e-2, history_size=8, max_iter=5)

        def closure():
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            if relax_inducing_points:
                self.Ksf = kernel(self.sparse_descriptors, self.full_descriptors, self.kernel_length)
                self.Kss = kernel(self.sparse_descriptors, self.sparse_descriptors, self.kernel_length)
            self.__update_Lss_Sigma_alpha()
            self.__compute_negative_log_marginal_likelihood()
            if self._nlml.requires_grad:
                self._nlml.backward()
            return self._nlml

        for param in params:
            param.requires_grad_()
        counter = 0
        closure()
        print(-self._nlml.item(), self.outputscale, self.noise)
        d_nlml = np.inf
        prev_nlml = self._nlml.item()
        while np.abs(d_nlml / prev_nlml) > rtol:
            counter += 1
            self.optimizer.step(closure)
            this_nlml = self._nlml.item()
            d_nlml = np.abs(this_nlml - prev_nlml)
            print(-self._nlml.item(), self.outputscale, self.noise)
            prev_nlml = this_nlml

        for param in params:
            param.requires_grad_(False)

        self.Lambda_inv = self.Lambda_inv.detach()
        self.Lss = self.Lss.detach()
        self.Sigma = self.Sigma.detach()
        self.alpha = self.alpha.detach()
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
                Kff = outputscale * kernel(full_set, full_set,
                                           self.kernel_length, diag=True)
                if update:  # do not recompute if not full set update
                    Ksf = outputscale * kernel(self.sparse_descriptors, full_set,
                                               self.kernel_length)
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
