import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Определяем переменную величину
frames = 500
seconds_in_year = 365 * 24 * 60 * 60
years = 2
t = np.linspace(0, years*seconds_in_year, frames)

# Определяем функцию для системы диф. уравнений
def move_func(s, t):
    (x1, v_x1, y1, v_y1) = s

    dxdt1 = v_x1
    dv_xdt1 = - G * M * x1 / (x1**2 + y1**2)**1.5
    dydt1 = v_y1
    dv_ydt1 = - G * M * y1 / (x1**2 + y1**2)**1.5
  
    return (dxdt1, dv_xdt1, dydt1, dv_ydt1)

# Определяем начальные значения и параметры
G = 6.67 * 10**(-11)
M = 1.998 * 10**(30)
ae = 149 * 10**9
edge = 5 * ae
e = 0.2

def Ve(a):
    return np.sqrt((G * M) / a * (2 / (1 - e) - 1 ))

# x0 = 0.5 * ae
# v_x0 = 0
# y0 = 0
# v_y0 = - Ve(0.5*ae, ae)

alpha1 = 75
a1 = ae * 0.5
x01 = a1 * np.cos(np.deg2rad(alpha1))
v_x01 = - Ve(a1) * np.sin(np.deg2rad(alpha1))
y01 = a1 *  np.sin(np.deg2rad(alpha1))
v_y01 = Ve(a1) * np.cos(np.deg2rad(alpha1))


s0 = (x01, v_x01, y01, v_y01)

sol = odeint(move_func, s0, t)

# Строим решение в виде графика и анимируем
fig, ax = plt.subplots()

number_points = 1
points = []
points_lines = []

for i in range(number_points):
    points.append(plt.plot([], [], 'o', color='r'))
    points_lines.append(plt.plot([], [], '-', color='r'))

def animate(i):
    for j in range(number_points):
        points[j][0].set_data([sol[i, 4 * j]], [sol[i, 4 * j + 2]])
        points_lines[j][0].set_data(sol[:i, 4 * j], sol[:i, 4 * j + 2])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

plt.axis('equal')
ax.set_xlim(-edge, edge)
ax.set_ylim(-edge, edge)

plt.plot([0], [0], 'o', color='r', ms=20)

ani.save('onq_body.gif', writer='pillow')