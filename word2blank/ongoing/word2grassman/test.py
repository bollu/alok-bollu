import numpy as np
from numpy import *
from numpy.linalg import *

def align_subspace(target, N, P):
  current = np.random.rand(N,P)
  grad = np.zeros((N,P))
  (tu, ts, tvtranspose) = np.linalg.svd(target)
  Q_t, R_t = np.linalg.qr(target)
  #while True:
  for x in range(500):
    # loss = sin(theta_max)
    Q_x, R_x = np.linalg.qr(current) 
    U, theta, vtrans = np.linalg.svd(Q_x.T@Q_t)
    V = vtrans.T
    index = np.where(theta == np.amin(theta))
    index = index[0][0]
    loss = np.sin(np.arccos(theta[index]))
    #if np.abs(loss) < 1e-2:
    # break
    print("loss: %4.2f | subspace:\n%s" % (loss, Q_x))
    for i in range(N):
      for j in range(P):
        grad[i][j] = 0.0
        sum = 0.0
        for l in range(P):
          sum += Q_t[i][l]*V[l][index]  
        grad[i][j] += -1*np.cos(np.arccos(theta[index]))*np.sqrt(1-(np.cos(np.arccos(theta[index]))*np.cos(np.arccos(theta[index]))))*U[j][index]*sum
        #np.sin(np.arccos(theta[index]))*
    current = current + grad*0.05

if __name__ == "__main__":
  align_subspace(np.asarray([[0, 0],[1, 0], [0,1]]), 3, 2)
  #align_subspace(np.asarray([[0,0],[1,0],[0,1]]), 3, 2)
  # align_subspace(np.asarray([[1, 0, 0], [0, 0, 1]], 3, 2)