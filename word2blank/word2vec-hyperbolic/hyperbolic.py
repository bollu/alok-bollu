#!/usr/bin/env python3
import math
from math import *

def scale(f, u, n): return [f * u[i] for i in range(n)]

# hyperbolic dot product
def hdot(u, v, n):
    f = 0
    for i in range(n-1): f += u[i] * v[i]
    f -= u[n-1]*v[n-1]
    return f

# exponential map
def exp_eq3(p, v, n):
    l = 0
    for i in range(n): l+= v[i] *v[i]
    l = sqrt(l)

    cl = cosh(l); sl = sinh(l)
    out = [cl*p[i] + sl * v[i] / l for i in range(n)]
    return out


# eqn 2: geodesic distance in local coordinates
def dist_eq2(u, v, n):
    dot = 0
    for i in range(n-1): dot += u[i] * v[i]
    dot -= u[n-1]*v[n-1]
    print("\t-dot: %4.2f" % dot)
    return acosh(dot)

# derivative of distance
def der_dist_eq4(u, v, n):
    d = hdot(u, v, n)
    c =  -1.0 / math.sqrt(d*d - 1)
    return [c * v[i] for i in range(n)]

# convert from ambient to tangent space of
# hyperbolic space
def ambient2tangent_eq5(p, gradambient, n):
    d = hdot(p, gradambient, n)
    return [gradambient[i] + d*p[i] for i in range(n)]

x = [0.5, 0.3]
y = [4, 0.4]
N = 2
LEARNINGRATE = 1e-4
for i in range(80000):
    # (4 - d)^2
    # -2d' * (4 - d)
    d = dist_eq2(x, y, N) 
    print ("%epoch: s | dist: %4.2f | x: %s --> y:%s" % (i, d, x, y))
    htangent = ambient2tangent_eq5(x, scale((4 - d), der_dist_eq4(x, y, N), N), N)
    x = exp_eq3(x, scale(-LEARNINGRATE, htangent, N), N)

