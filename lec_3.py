import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Определяем переменную величину
frames = 800
seconds_in_year = 365 * 24 * 60 * 60
years = 3      
t = np.linspace(0, years*seconds_in_year, frames)

# Определяем функцию для системы диф. уравнений
def move_func(s, t):
    (xc1, v_xc1, yc1, v_yc1,
     xc19, v_xc19, yc19, v_yc19) = s

    dxdtc1 = v_xc1
    dv_xdtc1 = - G * M * xc1 / (xc1 + yc1)**1.5
    dydtc1 = v_yc1
    dv_ydtc1 = - G * M * yc1 / (xc1 + yc1)**1.5
    
    dxdtc19 = v_xc19
    dv_xdtc19 = - G * M * xc19 / (xc19 + yc19)**1.5
    dydtc19 = v_yc19
    dv_ydtc19 = - G * M * yc19 / (xc19 + yc19)**1.5
  
    return (dxdtc1, dv_xdtc1, dydtc1, dv_ydtc1, 
            dxdtc19, dv_xdtc19, dydtc19, dv_ydtc19)

# Определяем начальные значения и параметры
G = 6.67 * 10**(-11)
M = 1.998 * 10**(30)
ae = 149 * 10**9
edge = 5 * ae
e = 0.2

def Ve(a):
    return np.sqrt((G * M) / a * (2 / (1 - e) - 1 ))

center_radius1 = 0.5 * ae
angular_speed = 2 * np.pi / (0.5 * seconds_in_year)

angle1 = 0
xc1 = center_radius1 * np.cos(np.deg2rad(angle1))
yc1 = center_radius1 * np.sin(np.deg2rad(angle1))
v_xc1 = -angular_speed * yc1
v_yc1 = angular_speed * xc1

center_radius2 = 0.3 * ae

angle19 = 1
xc19 = center_radius2 * np.cos(np.deg2rad(angle19))
yc19 = center_radius2 * np.sin(np.deg2rad(angle19))
v_xc19 = -angular_speed * yc19
v_yc19 = angular_speed * xc19


s0 = (xc1, v_xc1, yc1, v_yc1,
      xc19, v_xc19, yc19, v_yc19)

sol = odeint(move_func, s0, t)

# Строим решение в виде графика и анимируем
fig, ax = plt.subplots()

number_points = 1
points = []
points_lines = []

for i in range(number_points):
    points.append(plt.plot([], [], 'o', color='r'))
    # points_lines.append(plt.plot([], [], '-', color='r'))

def animate(i):
    for j in range(number_points):
        points[j][0].set_data([sol[i, 4 * j]], [sol[i, 4 * j + 2]])
        # points_lines[j][0].set_data(sol[:i, 4 * j], sol[:i, 4 * j + 2])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

plt.axis('equal')
ax.set_xlim(-edge, edge)
ax.set_ylim(-edge, edge)

# plt.plot([0], [0], 'o', color='r', ms=20)

ani.save('onq_body.gif', writer='pillow')