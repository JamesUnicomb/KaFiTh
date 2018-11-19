import time

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from KalmanFilters import BrownianFilter

bf = BrownianFilter(state       = 'x1 x2',
                    measurement = 'z1 z2')

q = 3.0
r = 0.1

Q = [[q * q,   0.0],
     [  0.0, q * q]]
R = [[r * r,   0.0],
     [  0.0, r * r]]

est_x = [[0.0,0.0]]
est_p = [[[10.0,  0.0],
          [ 0.0, 10.0]]]

timedelta = 0.01

gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[:, 0])
ax2 = plt.subplot(gs[0, 1:])
ax3 = plt.subplot(gs[1, 1:], sharey=ax2)

t = np.arange(0.0, 1.0, timedelta)
b = np.random.multivariate_normal([0.0, 0.0], timedelta * np.array(Q), size=len(t))
b = b.cumsum(axis=0)

ax1.plot(b[:,0], b[:,1], 'k')

b += np.random.multivariate_normal([0.0, 0.0], np.array(R), size=len(t)).reshape(b.shape)

ax1.scatter(b[:,0], b[:,1], c='k', marker='x', alpha=0.2)

start_time = time.time()

x_ = est_x[0]
p_ = est_p[0]

for m in b:
    x_, p_ = bf(x_, p_, m, Q, R, timedelta)
    est_x.append(x_)
    est_p.append(p_)
print 'The EKF took %.5fs per step' % ((time.time() - start_time)/len(b))

ax1.plot(np.array(est_x)[:,0], np.array(est_x)[:,1], c='C0')


ax2.plot(t,np.array(est_x)[1:,0] - b[:,0])
ax2.plot(t,+ 2.0 * np.sqrt(np.array(est_p, dtype=np.float32)[1:,0,0]), c='C0')
ax2.plot(t,- 2.0 * np.sqrt(np.array(est_p, dtype=np.float32)[1:,0,0]), c='C0')

ax3.plot(t,np.array(est_x)[1:,1] - b[:,1])
ax3.plot(t,+ 2.0 * np.sqrt(np.array(est_p, dtype=np.float32)[1:,1,1]), c='C0')
ax3.plot(t,- 2.0 * np.sqrt(np.array(est_p, dtype=np.float32)[1:,1,1]), c='C0')

plt.show()
