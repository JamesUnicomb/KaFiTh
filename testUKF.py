import numpy as np
from KalmanFilters import AutoRegressiveUnscentedKalmanFilter
from sklearn.datasets import make_spd_matrix

N = 10

ukf = AutoRegressiveUnscentedKalmanFilter(N)
x = np.arange(N)
P = make_spd_matrix(N) if 0 else np.eye(N)
mu_, x_, s_ = ukf.test_f(x, P)

print mu_
print np.round(np.cov(x_),3)
print np.round(s_,3)
