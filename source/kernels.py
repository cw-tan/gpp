import torch


class Kernel(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.kernel_hyperparameters = None

    def get_duplicate_ids(self, x1, x2, tol=1e-8):
        """
        Get indices of duplicate points based on a kernel function with output range [0,1]
        within given tolerance. The duplicate indices correspond to that of x2 (i.e. we
        keep x1 as is and remove the duplicate indices from x2).

        Args:
            x1, x2 (torch.Tensor): M × D tensors where M is the number of descriptor vectors
                                   and D is the dimensionality
            tol (float)          : how close the entries of K(x1, x2) are to 1
        """
        if x1.shape[0] > x2.shape[0]:
            x_larger, x_smaller = x1, x2
        else:
            x_larger, x_smaller = x2, x1
        triu_ids = torch.triu_indices(x_smaller.shape[0], x_larger.shape[0],
                                      offset=int(torch.equal(x1, x2)),
                                      device=self.device)
        K = self(x_smaller, x_larger)
        duplicate_triu_ids = torch.nonzero((1 - K[triu_ids[0], triu_ids[1]]) < tol, as_tuple=True)[0]
        if x1.shape[0] > x2.shape[0]:
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
        mask = torch.ones(x2.shape[0], dtype=torch.bool, device=x1.device)
        mask[ids_to_remove] = False
        all_ids = torch.arange(x2.shape[0], device=x1.device)
        ids_to_keep = all_ids[mask]
        return x2[ids_to_keep, :]


class SquaredExponentialKernel(Kernel):
    """
    Squared exponential (i.e. Gaussian) kernel.
    """
    def __init__(self, lengthscale=1.0, device='cpu'):
        super().__init__(device=device)
        self.lengthscale = torch.tensor([lengthscale], dtype=torch.float64, device=self.device)
        self.kernel_hyperparameters = [self.lengthscale]

    def forward(self, x1, x2, diag=False):
        """
        x1, x2 (torch.Tensor): N/M × D tensors where N/M is the number of descriptor vectors
                               and D is the dimensionality
        diag (bool)          : whether to only evaluate diagonal components
        """
        assert torch.is_tensor(x1) and torch.is_tensor(x2), 'x1 and x2 must be torch.Tensor'
        assert len(x1.shape) == 2 and len(x2.shape) == 2, 'x1 and x2 must be 2D tensors'
        assert x1.shape[1] == x2.shape[1], 'vector dimensions of x1 and x2 are incompatible'

        if not diag:
            aux_x1 = x1.unsqueeze(1)  # shape [N, 1, D] to broadcast with x2 with shape [M, D]
        else:
            assert x1.shape[0] == x2.shape[0], 'diag = True requires same dims'
            aux_x1 = x1
        scalar_form = torch.sum((aux_x1 - x2).pow(2), dim=-1)  # [N, M]
        return torch.exp(-0.5 * scalar_form / self.lengthscale.pow(2))
