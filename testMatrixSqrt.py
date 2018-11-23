import time
import numpy as np
from KalmanFilters import MatrixSqrt, sqrtm
from sklearn.datasets import make_spd_matrix
import scipy.linalg

sqrtm = sqrtm()


m = []
N = 1000
M = 40

for k in range(N):
    m.append(make_spd_matrix(M))

t0 = time.time()
error = []
for A in m:
    B = np.matrix(sqrtm.sqrtm(A))
    error.append(np.mean(np.square(B * B - A)))


print 'took ', (time.time() - t0)/N, ' for theano matrix square root with MSE ', np.mean(error)


t0 = time.time()
error = []
for A in m:
    B = np.matrix(scipy.linalg.sqrtm(A))
    error.append(np.mean(np.square(B * B - A)))


print 'took ', (time.time() - t0)/N, ' for scipy matrix square root with MSE ', np.mean(error)
