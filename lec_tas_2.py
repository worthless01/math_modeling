import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t = np.arange (0, 4.1, 0.1)

i_0 = 1000
k = 0.08

def bacteria(i, t):
    v = -k * i * t
    return v

i = odeint(bacteria, i_0, t)

plt.plot(t, i[:, 0], label = 'инвестиции')

plt.xlabel('время, годы')
plt.ylabel('инвестиции (ден. ед)')
plt.title('уменьшение инвестиций')
plt.legend()

plt.savefig('task_2.png')