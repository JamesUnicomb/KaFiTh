import time
import numpy as np
from KalmanFilters import AutoRegressiveUnscentedKalmanFilter
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt

t,b = np.loadtxt('mgdata.txt',delimiter=',')[::10].T
dt = np.mean(np.diff(t))

K = 50
S = int(0.5 * len(b))
E = -int(0.25 * len(b))

ukf = AutoRegressiveUnscentedKalmanFilter(K,
                                          num_units=64,
                                          eps=5e-3)
ukf.fit(b[:S])

q = 0.125
r = 0.125

Q = [[q * q]]
R = [[r * r]]

b_ = b + r * np.random.randn(len(b))
b_[E:] = np.nan


ukf_x = np.zeros(K)
ukf_p = np.eye(K)

ukf_est_x = []
ukf_est_p = []

for j in range(len(b)):
    ukf_x, ukf_p = ukf(ukf_x, ukf_p, [b_[j]], Q, R, dt)
    ukf_x = ukf_x.reshape(K)
    ukf_est_x.append(ukf_x[0])
    ukf_est_p.append(ukf_p[0,0])

ukf_est_x = np.array(ukf_est_x)
ukf_est_p = np.array(ukf_est_p)


plt.plot(t,b,'k--')
plt.scatter(t, b_, c='k', marker='x', alpha=0.2, label='noisy')
plt.plot(t,ukf_est_x,c='C0')
plt.fill_between(t,
                 ukf_est_x + 2.0 * np.sqrt(ukf_est_p),
                 ukf_est_x - 2.0 * np.sqrt(ukf_est_p),
                 color='C0', alpha=0.2)
plt.xlim([1400,1900])
plt.show()
