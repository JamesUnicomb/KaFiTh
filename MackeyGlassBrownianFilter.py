import time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from KalmanFilters import BrownianFilter

bf = BrownianFilter()

q = 0.1
r = 0.25

Q = [[q * q]]
R = [[r * r]]

est_x = [[0.0]]
est_p = [[[10.0]]]

t,b = np.loadtxt('mgdata.txt', delimiter=',')[::10].T
timedelta = np.mean(np.diff(t))

plt.figure(figsize=(12,6))
plt.subplot(211)

plt.plot(t,b,'k--',label='clean')

gt = b.copy()

b += np.random.multivariate_normal([0.0], np.array(R), size=len(t)).reshape(-1)

plt.scatter(t,b,c='k', marker='x', alpha=0.2, label='noisy')

start_time = time.time()

x_ = est_x[0]
p_ = est_p[0]

for m in b:
    x_, p_ = bf(x_, p_, [m], Q, R, timedelta)
    est_x.append(x_)
    est_p.append(p_)
print 'The EKF took %.5fs per step' % ((time.time() - start_time)/len(b))

plt.plot(t,np.array(est_x).reshape(-1)[1:], label='Kalman Filter')
plt.fill_between(t,
                 np.array(est_x).reshape(-1)[1:] + 2.0 * np.sqrt(np.array(est_p, dtype=np.float32)[1:,0,0]),
                 np.array(est_x).reshape(-1)[1:] - 2.0 * np.sqrt(np.array(est_p, dtype=np.float32)[1:,0,0]),
                 color='C0', alpha=0.2, label='2 sigma Error Bounds')
plt.legend(loc='upper right')
plt.xlim([400,700])


plt.subplot(212)
plt.plot(t,np.square(np.array(est_x).reshape(-1)[1:] - gt), label='Kalman Filter MSE')
plt.fill_between(t,
                 4.0 * np.array(est_p, dtype=np.float32)[1:,0,0],
                 0.0,
                 color='C0', alpha=0.2, label='2 sigma Error Bound')
plt.xlim([400,700])
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Results/MackeyGlassBrownianFilter.png',
            dpi=400)
plt.show()
