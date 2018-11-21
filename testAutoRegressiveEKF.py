import numpy as np
import matplotlib.pyplot as plt
from KalmanFilters import AutoRegressiveExtendedKalmanFilter

t,b = np.loadtxt('mgdata.txt',delimiter=',')[::10].T
dt = np.mean(np.diff(t))

K = 10
S = int(0.5 * len(b))
E = -int(0.25 * len(b))
ekf = AutoRegressiveExtendedKalmanFilter(K, eps=1e-2)
ekf.fit(b[:S])

q = 0.175
r = 0.125

Q = [[q * q]]
R = [[r * r]]

b_ = b + r * np.random.randn(len(b))

x = np.zeros(K)
p = np.eye(K)

est_x = []
est_p = []

for j in range(len(b)):
    x, p = ekf(x, p, [b_[j]], Q, R, dt)

    est_x.append(x[0])
    est_p.append(p[0,0])

est_x = np.array(est_x)
est_p = np.array(est_p)

plt.figure(figsize=(12,3))
plt.plot(t, b, 'k--', label='clean Mackey-Glass Series')
plt.scatter(t, b_, c='k', marker='x', alpha=0.2, label='noisy')
plt.plot(t, est_x, label='AREKF Estimate')
plt.fill_between(t,
                 est_x + 2.0 * np.sqrt(est_p),
                 est_x - 2.0 * np.sqrt(est_p),
                 color='C0', alpha=0.2, label='2 sigma Error Bounds')
plt.xlim([400,700])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Results/AutoRegressiveEKF.png',
            dpi=400)
plt.show()
