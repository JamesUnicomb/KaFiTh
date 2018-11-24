import time
import numpy as np
from KalmanFilters import AutoRegressiveUnscentedKalmanFilter
from sklearn.datasets import make_spd_matrix

N = 12

ukf = AutoRegressiveUnscentedKalmanFilter(N)
x = np.arange(N)
P = make_spd_matrix(N) if 0 else np.eye(N)
# 
# t0 = time.time()
# for j in range(1000):
#     mu_, x_, s_ = ukf.test_f(x, P)
#
# print (time.time() - t0) / 1000
#
# # print mu_
# # print np.round(np.cov(x_),3)
# # print np.round(s_,3)
# print x_
