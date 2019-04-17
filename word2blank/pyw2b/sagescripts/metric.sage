#!/usr/bin/env sage
M = Manifold(3, 'M', structure='Riemannian', start_index=1)
X.<x,y,z> = M.chart()
g = M.metric()
g[1,1], g[2,2], g[3,3] = 1, 1, 1
print(g.display())
