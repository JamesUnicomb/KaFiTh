import numpy as np
import matplotlib.pyplot as plt
from KalmanFilters import AutoRegressiveModel

t,b = np.loadtxt('mgdata.txt',delimiter=',')[::10].T
dt = np.mean(np.diff(t))

K = 40
S = int(0.5 * len(b))
E = -int(0.25 * len(b))
ar = AutoRegressiveModel(K, eps=1e-3)

print ar.fit(b[:S])

test = np.array([b[S:E][i:i-K] for i in range(K)]).T
res  = list(ar.f(test).reshape(-1))
tl   = list(t[:E])

for k in range(-E):
    res += [ar.f(np.array(res[-K:]).reshape(-1,K))[0,0]]
    tl  += [tl[-1] + dt]


plt.figure(figsize=(12,3))
plt.plot(t,b,label='clean Mackey-Glass Series')
plt.plot(tl[S+K:E],res[:E],label='One Step Prediction')
plt.plot(tl[E:],res[E:],label='Multi Step Prediction')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Results/AutoRegressiveModel.png',
            dpi=400)
plt.show()
