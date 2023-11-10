import torch
import numpy as np

def kernel(x1, x2, noise=1, l=1):
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

  x1_mat = torch.transpose(x1_mat.unsqueeze(2).expand(x1_mat.shape + (x2_mat.shape[1],)), 0, 1)  # [m, d, n]
  # the only thing different for other kernels is the last two lines
  scalar_form = torch.sum((x1_mat - x2_mat).pow(2), dim=1)  # [m, n]
  return noise* noise * torch.exp(-0.5 * scalar_form / (l * l))

  #norm = torch.sum(x1_mat.pow(2), dim=1) * torch.sum(x2_mat.pow(2), dim=0)
  #return noise * noise * torch.sum(x1_mat  * x2_mat , dim=1).pow(2) / norm

class GaussianProcess():
    """
    TODO: device manipulation
    """
    def __init__(self, descriptor_dim):
        self.descriptor_dim = descriptor_dim
        self.training_descriptors = torch.empty((descriptor_dim, 0))
        self.training_outputs = torch.empty((0,))

        # hyperparameters
        self.kernel_noise = torch.tensor([1], dtype=torch.float64)
        self.kernel_length = torch.tensor([1], dtype=torch.float64)
        self.model_noise = torch.tensor([0.01], dtype=torch.float64)
        self.optimizer = torch.optim.Rprop([self.model_noise, self.kernel_noise, self.kernel_length])

        # trained model parameters (precomputed during model updates for faster prediction)
        self.L = None
        self.alpha = None  # for predicting mean

    def update_model(self, x_train, y_train):
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
        self.training_descriptors = torch.cat((self.training_descriptors, torch.atleast_2d(x_train)), dim=1)
        self.training_outputs = torch.cat((self.training_outputs, y_train))
        self.update_L_alpha()

    def update_L_alpha(self):
        Kff = kernel(self.training_descriptors, self.training_descriptors,
                     self.kernel_noise, self.kernel_length)
        # Cholesky decomposition - n³/3 flops, but we only need to do it when adding new training points
        self.L = torch.linalg.cholesky(Kff + self.model_noise**2 * torch.eye(self.training_descriptors.shape[1]))
        # solving by forward and back substitution with Cholesky decomposed L - 2n² flops
        self.alpha = torch.cholesky_solve(self.training_outputs.expand(1, len(self.training_outputs)).T, self.L, upper=False)[:,0]

    def get_predictions(self, x_test, mean_var=[True, True]):
        """
        Get predictions of GP model with a set of testing vectors.

        Args:
          x_test (torch.Tensor): d × p tensor where p is the number of x vectors and d is the
                                  dimensionality of the x descriptor vectors
        """
        Kft = kernel(self.training_descriptors, x_test,
                     self.kernel_noise, self.kernel_length)
        predictions = []
        if mean_var[0]:
            predictions.append(Kft.T @ self.alpha)  # 2pn flops
        if mean_var[1]:
            Ktt = kernel(x_test, x_test,
                        self.kernel_noise, self.kernel_length)
            # do forward substitutioh p times - pn² flops
            v = torch.linalg.solve_triangular(self.L, Kft, upper=False)
            predictions.append(torch.abs((Ktt - v.T @ v).diag()))
        return predictions

    def get_likelihood(self):
        return -0.5 * self.training_outputs.unsqueeze(0) @ self.alpha - torch.sum(self.L.diag()) \
               - self.training_descriptors.shape[1] * 0.5 * np.log(2 * np.pi)

    def optimize_hyperparameters(self, rtol=1e-4, relax_kernel_length=False):
        """
        Optimize hyperparameters
        """
        self.model_noise.requires_grad_()
        self.kernel_noise.requires_grad_()
        self.kernel_length.requires_grad_(relax_kernel_length)
        counter = 0
        self.optimizer.zero_grad()
        self.update_L_alpha()
        likelihood = self.get_likelihood()
        (-likelihood).backward()
        dlikelihood = np.inf
        prev_likelihood = likelihood.item()
        while np.abs(dlikelihood/prev_likelihood) > rtol:
            counter += 1
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.update_L_alpha()
            likelihood = self.get_likelihood()
            (-likelihood).backward()
            dlikelihood = np.abs(prev_likelihood - likelihood.item())
            prev_likelihood = likelihood.item()
        self.model_noise.requires_grad_(False)
        self.kernel_noise.requires_grad_(False)
        self.kernel_length.requires_grad_(False)
        self.L = self.L.detach()
        self.alpha = self.alpha.detach()
        return counter