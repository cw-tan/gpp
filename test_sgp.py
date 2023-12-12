import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source'))

from sgp import SparseGaussianProcess 
from kernels import SquaredExponentialKernel

device = 'cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def func(x):
    return torch.cos(0.5*x) - 0.3 * x + 0.1 * torch.exp(0.3 * x) + 0.05 * torch.rand(x.shape, dtype=torch.float64, device=device)

L = 10
x_test = torch.linspace(-L - 1, L + 1, 1000, dtype=torch.float64, device=device)

kernel = SquaredExponentialKernel(device=device)

#for p in kernel.parameters():
#    print(p)


SGP = SparseGaussianProcess(1, kernel, decomp_mode='v', sgp_mode='vfe', init_noise=0.05, device=device)

for n, p in SGP.named_parameters():
    print(n, p)

print(SGP.device)

print(SGP.outputscale, SGP.noise)

x_train = L * (2 * torch.rand(50, dtype=torch.float64, device=device) - 1)
y_train = func(x_train)
plt.plot(x_train.cpu(), y_train.cpu(), 'ko')

x_sparse = torch.atleast_2d(x_train[torch.randperm(len(x_train))[:10]])
x_sparse = kernel.remove_duplicates(x_sparse, x_sparse, tol=1e-7)
SGP.update_model(torch.atleast_2d(x_train), y_train, x_sparse)
plt.plot(x_train.cpu(), y_train.cpu(), 'ko', label='Training Points')


for i in range(5):
    x_train = L * (2 * torch.rand(20, dtype=torch.float64, device=device) - 1)
    y_train = func(x_train)
    plt.plot(x_train.cpu(), y_train.cpu(), 'ko')

    x_sparse_new = torch.atleast_2d(x_train[torch.randperm(len(x_train))[:5]])
    x_sparse_new = kernel.remove_duplicates(x_sparse_new, x_sparse_new, tol=1e-7)
    x_sparse_new = kernel.remove_duplicates(SGP.sparse_descriptors, x_sparse_new, tol=1e-7)
    SGP.update_model(torch.atleast_2d(x_train), y_train, x_sparse_new)


x_sparse = SGP.sparse_descriptors
plt.plot(x_sparse[0,:].cpu(), torch.full((x_sparse.shape[1], ), -3), 'b^', label='Initial Inducing Points', markersize=15)

steps = SGP.optimize_hyperparameters(rtol=1e-4, relax_inducing_points=True, relax_kernel_params=False)
#print(steps)
steps = SGP.optimize_hyperparameters(rtol=1e-4, relax_inducing_points=False, relax_kernel_params=True)
print(SGP.kernel.lengthscale)

x_sparse = SGP.sparse_descriptors
plt.plot(x_sparse[0,:].cpu(), torch.full((x_sparse.shape[1], ), -2.5), 'g^', label='Optimized Inducing Points', markersize=15)


mean, var = SGP(torch.atleast_2d(x_test), mean_var=[True, True], include_noise=True)

plt.fill_between(x_test.cpu(), mean.cpu() - torch.sqrt(var.cpu()), mean.cpu() + torch.sqrt(var.cpu()),
                 color='lightcoral', alpha=0.4, label=r'$\mu(x)\pm\sigma(x)$')
plt.plot(x_test.cpu(), mean.cpu(), '--', color='red', label=r'$\mu(x)$')


plt.ylabel(r'$y$')
plt.xlabel(r'$x$')
"""
plt.legend(*(
    [ x[i] for i in [0,3,1,2] ]
    for x in plt.gca().get_legend_handles_labels()
), handletextpad=0.75, loc='best', frameon=False)
"""
plt.legend(frameon=False)

plt.xlim([-L - 1, L + 1])
plt.show()