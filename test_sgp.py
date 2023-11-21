import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source'))

from sgp import SparseGaussianProcess 
from kernels import SquaredExponentialKernel

def func(x):
    return torch.cos(0.5*x) - 0.3 * x + 0.1 * torch.exp(0.3 * x) + 0.5 * torch.rand(x.shape)

L = 10
x_test = torch.linspace(-L - 1, L + 1, 1000, dtype=torch.float64)

kernel = SquaredExponentialKernel()
SGP = SparseGaussianProcess(1, kernel, invert_mode='v', sgp_mode='vfe')

print(SGP.outputscale, SGP.noise)

#plt.plot(x_test, func(x_test), '-k', label=r'$f(x)$')
x_train = L * (2 * torch.rand(100, dtype=torch.float64) - 1)
y_train = func(x_train)
plt.plot(x_train, y_train, 'ko')

x_sparse = torch.atleast_2d(x_train[torch.randperm(len(x_train))[:10]])
x_sparse = kernel.remove_duplicates(x_sparse, x_sparse, tol=1e-7)
SGP.update_model(torch.atleast_2d(x_train), y_train, x_sparse)
plt.plot(x_train, y_train, 'ko', label='Training Points')
plt.plot(x_sparse[0,:], torch.full((x_sparse.shape[1], ), -3), 'b^', label='Initial Inducing Points', markersize=15)

"""
for i in range(10):
    x_train = L * (2 * torch.rand(10, dtype=torch.float64) - 1)
    
    x_sparse_new = torch.atleast_2d(x_train[torch.randperm(len(x_train))[:2]])
    x_sparse_new = kernel.remove_duplicates(x_sparse_new, x_sparse_new, tol=1e-7)
    x_sparse_new = kernel.remove_duplicates(SGP.sparse_descriptors, x_sparse_new, tol=1e-7)

    SGP.update_sparse_set(x_sparse_new)
    plt.plot(x_sparse_new[0,:], func(x_sparse_new[0,:]), 'g+', markersize=15)

    y_train = func(x_train)
    SGP.update_full_set(torch.atleast_2d(x_train), y_train)
    plt.plot(x_train, y_train, 'ko')
"""

steps = SGP.optimize_hyperparameters(relax_inducing_points=True, relax_kernel_params=True)
print(steps)

x_sparse = SGP.sparse_descriptors
plt.plot(x_sparse[0,:], torch.full((x_sparse.shape[1], ), -2.5), 'g^', label='Optimized Inducing Points', markersize=15)


mean, var = SGP.get_predictions(x_test, mean_var=[True, True], include_noise=True)

plt.fill_between(x_test, mean - torch.sqrt(var), mean + torch.sqrt(var), color='lightcoral', alpha=0.4, label=r'$\mu(x)\pm\sigma(x)$')
plt.plot(x_test, mean, '--', color='red', label=r'$\mu(x)$')


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