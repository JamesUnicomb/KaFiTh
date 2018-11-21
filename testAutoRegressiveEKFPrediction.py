import numpy as np
import matplotlib.pyplot as plt
from KalmanFilters import BrownianFilter, AutoRegressiveExtendedKalmanFilter

t,b = np.loadtxt('mgdata.txt',delimiter=',')[::10].T
dt = np.mean(np.diff(t))

K = 50
S = int(0.5 * len(b))
E = -int(0.25 * len(b))

bf = BrownianFilter()
ekf = AutoRegressiveExtendedKalmanFilter(K,
                                         num_units=64,
                                         eps=5e-3)
ekf.fit(b[:S])

q = 0.15
r = 0.125

Q = [[q * q]]
R = [[r * r]]

b_ = b + r * np.random.randn(len(b))

bf_x = np.zeros(1)
bf_p = np.eye(1)

ekf_x = np.zeros(K)
ekf_p = np.eye(K)


bf_est_x = []
bf_est_p = []

ekf_est_x = []
ekf_est_p = []

for j in range(len(b)):
    if j < len(b) + E:
        bf_x, bf_p   = bf(bf_x, bf_p, [b_[j]], Q, R, dt)
        ekf_x, ekf_p = ekf(ekf_x, ekf_p, [b_[j]], Q, R, dt)
    else:
        bf_x, bf_p   = bf(bf_x, bf_p, [np.nan], Q, R, dt)
        ekf_x, ekf_p = ekf(ekf_x, ekf_p, [np.nan], Q, R, dt)

    bf_est_x.append(bf_x[0])
    bf_est_p.append(bf_p[0,0])

    ekf_est_x.append(ekf_x[0])
    ekf_est_p.append(ekf_p[0,0])

bf_est_x = np.array(bf_est_x)
bf_est_p = np.array(bf_est_p)

ekf_est_x = np.array(ekf_est_x)
ekf_est_p = np.array(ekf_est_p)

plt.figure(figsize=(12,3))
plt.plot(t, b, 'k--', label='clean Mackey-Glass Series')
plt.scatter(t, b_, c='k', marker='x', alpha=0.2, label='noisy')
plt.plot(t, ekf_est_x, c='C0', label='AREKF Estimate')
plt.plot(t, bf_est_x, c='C1', label='BF Estimate')
plt.fill_between(t,
                 ekf_est_x + 2.0 * np.sqrt(ekf_est_p),
                 ekf_est_x - 2.0 * np.sqrt(ekf_est_p),
                 color='C0', alpha=0.2, label='2 sigma Error Bounds')
plt.fill_between(t,
                 bf_est_x + 2.0 * np.sqrt(bf_est_p),
                 bf_est_x - 2.0 * np.sqrt(bf_est_p),
                 color='C1', alpha=0.2, label='2 sigma Error Bounds')
plt.xlim([1400,1900])
plt.ylim([-0.8,2.5])
plt.legend(loc='upper right')
plt.savefig('Results/AutoRegressiveEKFPrediction.png',
            dpi=400)
plt.show()
