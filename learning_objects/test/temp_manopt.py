import torch
import autograd.numpy as np

import pymanopt
from pymanopt.manifolds import Rotations
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

# (1) Instantiate a manifold
manifold = Rotations(n=3, k=1)

# (2) Define the cost function (here using autograd.numpy)
# @pymanopt.function.Autograd
A = torch.rand(3, 3)
A.grad()
B =
def cost(X):
    return np.sum(X)

problem = Problem(manifold=manifold, cost=cost)

# (3) Instantiate a Pymanopt solver
solver = SteepestDescent()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)