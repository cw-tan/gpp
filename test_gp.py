import torch
import numpy as np
import matplotlib.pyplot as plt

from gp import GaussianProcess

def func(x):
  return torch.cos(0.5*x) - 0.3 * x + 0.1 * np.exp(0.3 * x)
  

L = 10
x_test = torch.linspace(-L - 1, L + 1, 1000, dtype=torch.float64)


GP = GaussianProcess(1)
x_trains = []
y_trains = []
for i in range(5):
    x_train = L * (2 * torch.rand(8, dtype=torch.float64) - 1)
    y_train = func(x_train)
    
    plt.plot(x_test, func(x_test), '-k', label=r'$f(x)$')
    x_trains.append(x_train)
    y_trains.append(y_train)

    GP.update_model(x_train, y_train)
    steps = GP.optimize_hyperparameters()
    print(steps)

    mean, var = GP.get_predictions(x_test, mean_var=[True, True])

    plt.fill_between(x_test, mean - torch.sqrt(var), mean + torch.sqrt(var), color='lightcoral', alpha=0.4, label=r'$\mu(x)\pm\sigma(x)$')
    plt.plot(x_test, mean, '--', color='red', label=r'$\mu(x)$')

    plt.plot(np.concatenate(x_trains), np.concatenate(y_trains), 'ko', label='Training Points')

    plt.ylabel(r'$y$')
    plt.xlabel(r'$x$')

    plt.legend(*(
        [ x[i] for i in [0,3,1,2] ]
        for x in plt.gca().get_legend_handles_labels()
    ), handletextpad=0.75, loc='best', frameon=False)

    plt.xlim([-L - 1, L + 1])
    plt.show()