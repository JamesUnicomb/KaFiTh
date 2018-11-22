import time
import numpy as np
from KalmanFilters import MatrixSqrt
from sklearn.datasets import make_spd_matrix

sqrtm = MatrixSqrt()

m = []
N = 1000
M = 40

for k in range(N):
    m.append(make_spd_matrix(M))

t0 = time.time()
for A in m:
    B = np.matrix(sqrtm(A))


print 'took ', (time.time() - t0)/N, ' for matrix square root.'
