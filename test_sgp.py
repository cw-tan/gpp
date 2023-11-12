import torch
import numpy as np
import matplotlib.pyplot as plt

from sgp import SparseGaussianProcess

def func(x):
  #return (torch.cos(0.5 * x) - 0.5 * torch.sin(2 * x) * torch.cos(x / 40)) * torch.exp(- 1e-4 * x* x)
  return torch.cos(0.5*x) - 0.3 * x + 0.1 * np.exp(0.3 * x)

L = 10
x_test = torch.linspace(-L - 1, L + 1, 1000, dtype=torch.float64)


SGP = SparseGaussianProcess(1, invert_mode='QR')

plt.plot(x_test, func(x_test), '-k', label=r'$f(x)$')
x_train = L * (2 * torch.rand(150, dtype=torch.float64) - 1)
y_train = func(x_train)

# use 25 % of full training points as sparse points
x_sparse = x_train[torch.randperm(len(x_train))[:(int(len(x_train) / 5))]]
#x_sparse[-1] = x_sparse[0] + 1e-7

SGP.update_model(x_train, y_train, x_sparse)

x_train = L * (2 * torch.rand(150, dtype=torch.float64) - 1)
y_train = func(x_train)
SGP.update_full_set(x_train, y_train)


x_sparse_new = L * (2 * torch.rand(20, dtype=torch.float64) - 1)
SGP.update_sparse_set(x_sparse_new)

#steps = SGP.optimize_hyperparameters(relax_kernel_length=False)
#print(steps)

mean, var = SGP.get_predictions(x_test, mean_var=[True, True], mode='dtc')

plt.fill_between(x_test, mean - torch.sqrt(var), mean + torch.sqrt(var), color='lightcoral', alpha=0.4, label=r'$\mu(x)\pm\sigma(x)$')
plt.plot(x_test, mean, '--', color='red', label=r'$\mu(x)$')

plt.plot(x_train, y_train, 'ko', label='Training Points')
plt.plot(x_sparse, func(x_sparse), 'g+', label='Sparse Points', markersize=15)


plt.ylabel(r'$y$')
plt.xlabel(r'$x$')

plt.legend(*(
    [ x[i] for i in [0,3,1,2] ]
    for x in plt.gca().get_legend_handles_labels()
), handletextpad=0.75, loc='best', frameon=False)

plt.xlim([-L - 1, L + 1])
plt.show()