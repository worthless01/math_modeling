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
    (x1, v_x1, y1, v_y1,
     x2, v_x2, y2, v_y2) = s

    # dxdt = v_x
    # dv_xdt = - G * M * x / (x**2 + y**2)**1.5
    # dydt = v_y
    # dv_ydt = - G * M * y / (x**2 + y**2)**1.5

    dxdt1 = v_x1
    dv_xdt1 = - G * M * x1 / (x1**2 + y1**2)**1.5
    dydt1 = v_y1
    dv_ydt1 = - G * M * y1 / (x1**2 + y1**2)**1.5

    dxdt2 = v_x2
    dv_xdt2 = - G * M * x2 / (x2**2 + y2**2)**1.5
    dydt2 = v_y2
    dv_ydt2 = - G * M * y2 / (x2**2 + y2**2)**1.5

    return (dxdt1, dv_xdt1, dydt1, dv_ydt1,
            dxdt2, dv_xdt2, dydt2, dv_ydt2)

# Определяем начальные значения и параметры
G = 6.67 * 10**(-11)
M = 1.998 * 10**(30)
ae = 200 * 10**9
edge = 2.5 * ae
e = 0.5

def Ve(r, a):
    return np.sqrt((G * M) / a * (2 / (1 - e) - 1 ))

# x0 = 0.5 * ae
# v_x0 = 0
# y0 = 0
# v_y0 = - Ve(0.5*ae, ae)

x01 = 0.5 * ae
v_x01 = 0
y01 = 0
v_y01 = - Ve(0.5*ae, ae)

x02 = 0.5 * ae *  np.cos(np.deg2rad(30))
v_x02 = - Ve(0.5*ae, ae) * np.sin(np.deg2rad(30))
y02 = - 0.5 * ae *  np.sin(np.deg2rad(30))
v_y02 = Ve(0.5*ae, ae) * np.cos(np.deg2rad(30))

s0 = (x01, v_x01, y01, v_y01,
      x02, v_x02, y02, v_y02)

sol = odeint(move_func, s0, t)

# Строим решение в виде графика и анимируем
fig, ax = plt.subplots()

number_points = 2
points = []
points_lines = []

for i in range(number_points):
    points.append(plt.plot([], [], 'o', color='r'))
    points_lines.append(plt.plot([], [], '-', color='r'))

def animate(i):
    for j in range(number_points):
        # Исправление: передаем точки как списки с одним элементом
        points[j][0].set_data([sol[i, 4 * j]], [sol[i, 4 * j + 2]])
        points_lines[j][0].set_data(sol[:i, 4 * j], sol[:i, 4 * j + 2])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

plt.axis('equal')
ax.set_xlim(-edge, edge)
ax.set_ylim(-edge, edge)

plt.plot([0], [0], 'o', color='r', ms=20)

ani.save('galaxy.gif', writer='pillow')
