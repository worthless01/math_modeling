import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t = np.arange (0, 10, 0.1)

N_0 = 10
k = 0.5

def bacteria(N, t):
    v = k * N
    return v


N = odeint(bacteria, N_0, t)

plt.plot(t, N[:, 0], label = 'рост бактерий')

plt.xlabel('время, часы')
plt.ylabel('количество бактерий')
plt.title('рост бактерий')
plt.legend()

plt.savefig('task_1.png')