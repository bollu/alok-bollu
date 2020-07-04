#!/usr/bin/env python3
import cvxpy as cp
import numpy as np

# Problem data.
# Glove eqn 18/8
m = 30
n = 20
np.random.seed(1)
X = np.random.randn(n, n)
fX = np.random.randn(n, n)
B = np.random.randn(n, n)

# Construct the problem.
W = cp.Variable((n, n))

objective = cp.Maximize(cp.prod(fX, cp.sum(W @ W.T + B - np.log(X))))
#objective = cp.Minimize(cp.sum_squares(A@x - b))
# constraints = [0 <= x, x <= 1]
contraints = [W >= 0]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)
