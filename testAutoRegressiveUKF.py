import time
import numpy as np
from KalmanFilters import AutoRegressiveUnscentedKalmanFilter,  AutoRegressiveExtendedKalmanFilter
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt

t,b = np.loadtxt('mgdata.txt',delimiter=',')[::10].T
dt = np.mean(np.diff(t))

K = 100
S = int(0.5 * len(b))
E = -int(0.25 * len(b))

ukf = AutoRegressiveUnscentedKalmanFilter(K,
                                          num_layers=2,
                                          num_units=64,
                                          eps=5e-3)
ukf.fit(b[:S])
ekf = AutoRegressiveExtendedKalmanFilter(K,
                                         num_layers=2,
                                         num_units=64,
                                         eps=5e-3)
ekf.fit(b[:S])

q = 0.15
r = 0.25

Q = [[q * q]]
R = [[r * r]]

b_ = b + r * np.random.randn(len(b))
b_[E:] = np.nan

ekf_x = np.zeros(K)
ekf_p = np.eye(K)

ukf_x = np.zeros(K)
ukf_p = np.eye(K)

ukf_est_x = []
ukf_est_p = []

ekf_est_x = []
ekf_est_p = []

for j in range(len(b)):
    ukf_x, ukf_p = ukf(ukf_x, ukf_p, [b_[j]], Q, R, dt)
    ekf_x, ekf_p = ekf(ekf_x, ekf_p, [b_[j]], Q, R, dt)

    ukf_x = ukf_x.reshape(K)
    ukf_est_x.append(ukf_x[0])
    ukf_est_p.append(ukf_p[0,0])

    ekf_est_x.append(ekf_x[0])
    ekf_est_p.append(ekf_p[0,0])


ukf_est_x = np.array(ukf_est_x)
ukf_est_p = np.array(ukf_est_p)

ekf_est_x = np.array(ekf_est_x)
ekf_est_p = np.array(ekf_est_p)

plt.figure(figsize=(12,6))
plt.subplot(211)
plt.plot(t,b,'k--',label='clean Mackey-Glass Series')
plt.scatter(t, b_, c='k', marker='x', alpha=0.2, label='noisy')
plt.plot(t,ekf_est_x,c='C0', label='AREKF Estimate')
plt.fill_between(t,
                 ekf_est_x + 2.0 * np.sqrt(ekf_est_p),
                 ekf_est_x - 2.0 * np.sqrt(ekf_est_p),
                 color='C0', alpha=0.2, label='2 sigma Error Bounds')
plt.plot(t,ukf_est_x,c='C2', label='ARUKF Estimate')
plt.fill_between(t,
                 ukf_est_x + 2.0 * np.sqrt(ukf_est_p),
                 ukf_est_x - 2.0 * np.sqrt(ukf_est_p),
                 color='C2', alpha=0.2, label='2 sigma Error Bounds')
plt.xlim([400,700])
plt.ylim([-0.8,2.5])
plt.legend(loc='upper right')


plt.subplot(212)
plt.plot(t,np.square(np.array(ekf_est_x).reshape(-1) - b), c='C0', label='AREKF MSE')
plt.plot(t,np.square(np.array(ukf_est_x).reshape(-1) - b), c='C2', label='ARUKF MSE')
plt.fill_between(t,
                 4.0 * np.array(ekf_est_p, dtype=np.float32),
                 0.0,
                 color='C0', alpha=0.2, label='2 sigma Error Bound')
plt.legend(loc='upper right')
plt.fill_between(t,
                 4.0 * np.array(ukf_est_p, dtype=np.float32),
                 0.0,
                 color='C2', alpha=0.2, label='2 sigma Error Bound')
plt.legend(loc='upper right')
plt.xlim([400,700])


plt.savefig('Results/AutoRegressiveUKF.png',
            dpi=400)
plt.show()
