import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t = np.arange (0, 10, 0.1)

a_0 = 3
m = 2
v_0 = 0
c = 0.5

def velocity(v, t):
    a = a_0 -(c / m) * v**2
    return a

v = odeint(velocity, v_0, t)

plt.plot(t, v[:, 0], label = 'скорость тела')

plt.xlabel('время, секунды')
plt.ylabel('скоростьт, м/с')
plt.title('движ тела с сопротивлением')
plt.legend()

plt.savefig('task_3.png')