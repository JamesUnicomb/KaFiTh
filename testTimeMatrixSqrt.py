import time
import numpy as np
from KalmanFilters import MatrixSqrt, sqrtm
from sklearn.datasets import make_spd_matrix
import scipy.linalg
import matplotlib.pyplot as plt

sqrtm = sqrtm()


m = []
N = 100

result = []

for M in np.square(np.arange(2,11)):
    for k in range(N):
        m.append(make_spd_matrix(M))

    Tt0 = time.time()
    error = []
    for A in m:
        B = np.matrix(sqrtm(A))
    Tt1 = time.time()


    St0 = time.time()
    error = []
    for A in m:
        B = np.matrix(scipy.linalg.sqrtm(A))
    St1 = time.time()

    result.append([M, (Tt1 - Tt0) / N, (St1 - St0) / N])

result = np.array(result)

ax = plt.subplot(111)
plt.plot(result[:,0], result[:,1], label='Theano (precompiled)')
plt.plot(result[:,0], result[:,2], label='SciPy')
plt.yscale('log')
plt.xlabel('N - Matrix Dimension (NxN)')
plt.ylabel('log Time (sec)')
plt.legend(loc='upper left')
plt.savefig('Results/MatrixSquareRootTiming.png', dpi=400)
plt.show()
