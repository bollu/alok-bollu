#- Get the principal angle working for matrices of our choice whose angles we know : 
# so x, y, x /\ y, and 0 (so we are talking about 2x2) (Code some basic stuff and check the validity of what we learned) 
import numpy as np
from numpy import *
from numpy.linalg import *

def align_subspace(target, N, P):
   #current = np.asarray([[0,0],[3,0],[0,10]])
   current = np.random.rand(N,P)
   grad = np.zeros((N,P))
   Q_t, R_t = np.linalg.qr(target)
   for x in range(500):
       # compute_loss
       # loss1 = \sum_i theta_i*theta_i
       Q_x, R_x = np.linalg.qr(current) 
       U, theta, vtrans = np.linalg.svd(Q_x.T@Q_t)
       V = vtrans.T
       loss = 0
       for cos_t in theta:
         loss += np.arccos(cos_t)**2
      #  if np.abs(loss) < 1e-2:
      #    break
       print("loss: %4.2f | subspace:\n%s" % (loss, Q_x))
       for i in range(N):
         for j in range(P):
           grad[i][j] = 0.0
           for cos_t in theta:
            index = np.where(theta == cos_t)
            index = index[0][0]
            sum = 0.0
            for l in range(P):
              sum += Q_t[i][l]*V[l][index]  
            grad[i][j] += -2*arccos(cos_t)*np.sqrt(1-(np.cos(np.arccos(cos_t))*np.cos(np.arccos(cos_t))))*U[j][index]*sum
       current = current + grad*0.05

if __name__ == "__main__":
  align_subspace(np.asarray([[0, 0],[1, 0], [0,1]]), 3, 2)
  #align_subspace(np.asarray([[0,0],[1,0],[0,1]]), 3, 2)
  # align_subspace(np.asarray([[1, 0, 0], [0, 0, 1]], 3, 2)

