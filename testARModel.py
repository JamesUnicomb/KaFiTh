import numpy as np
import matplotlib.pyplot as plt
from KalmanFilters import AutoRegressiveModel

t,b = np.loadtxt('mgdata.txt',delimiter=',')[::10].T
dt = np.mean(np.diff(t))

K = 50
S = 1000
E = -400
ar = AutoRegressiveModel(K, eps=1e-3)

print ar.fit(b[:S])

test = np.array([b[S:E][i:i-K] for i in range(K)]).T
res  = list(ar.f(test).reshape(-1))
tl   = list(t[:E])

for k in range(-E):
    res += [ar.f(np.array(res[-K:]).reshape(-1,K))[0,0]]
    tl  += [tl[-1] + dt]


plt.plot(t,b)
plt.plot(tl[S+K:E],res[:E])
plt.plot(tl[E:],res[E:])
plt.show()
