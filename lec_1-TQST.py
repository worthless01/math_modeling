import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Определяем переменную величину
frames = 500
seconds_in_year = 365 * 24 * 60 * 60
years = 3
t = np.linspace(0, years*seconds_in_year, frames)

# Определяем функцию для системы диф. уравнений
def move_func(s, t):
    (x1, v_x1, y1, v_y1,
     x2, v_x2, y2, v_y2,
     x3, v_x3, y3, v_y3,
     x4, v_x4, y4, v_y4,
     x5, v_x5, y5, v_y5,
     x6, v_x6, y6, v_y6,
     x7, v_x7, y7, v_y7,
     x8, v_x8, y8, v_y8,
     x9, v_x9, y9, v_y9,
     x10, v_x10, y10, v_y10,
     x11, v_x11, y11, v_y11) = s
    
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

    dxdt3 = v_x3
    dv_xdt3 = - G * M * x3 / (x3**2 + y3**2)**1.5
    dydt3 = v_y3
    dv_ydt3 = - G * M * y3 / (x3**2 + y3**2)**1.5

    dxdt4 = v_x4
    dv_xdt4 = - G * M * x4 / (x4**2 + y4**2)**1.5
    dydt4 = v_y4
    dv_ydt4 = - G * M * y4 / (x4**2 + y4**2)**1.5

    dxdt5 = v_x5
    dv_xdt5 = - G * M * x5 / (x5**2 + y5**2)**1.5
    dydt5 = v_y5
    dv_ydt5 = - G * M * y5 / (x5**2 + y5**2)**1.5

    dxdt6 = v_x6
    dv_xdt6 = - G * M * x6 / (x6**2 + y6**2)**1.5
    dydt6 = v_y6
    dv_ydt6 = - G * M * y6 / (x6**2 + y6**2)**1.5

    dxdt7 = v_x7
    dv_xdt7 = - G * M * x7 / (x7**2 + y7**2)**1.5
    dydt7 = v_y7
    dv_ydt7 = - G * M * y7 / (x7**2 + y7**2)**1.5

    dxdt8 = v_x8
    dv_xdt8 = - G * M * x8 / (x8**2 + y8**2)**1.5
    dydt8 = v_y8
    dv_ydt8 = - G * M * y8 / (x8**2 + y8**2)**1.5

    dxdt9 = v_x9
    dv_xdt9 = - G * M * x9 / (x9**2 + y9**2)**1.5
    dydt9 = v_y9
    dv_ydt9 = - G * M * y9 / (x9**2 + y9**2)**1.5

    dxdt10 = v_x10
    dv_xdt10 = - G * M * x10 / (x10**2 + y10**2)**1.5
    dydt10 = v_y10
    dv_ydt10 = - G * M * y10 / (x10**2 + y10**2)**1.5

    dxdt11 = v_x11
    dv_xdt11 = - G * M * x11 / (x11**2 + y11**2)**1.5
    dydt11 = v_y11
    dv_ydt11 = - G * M * y11 / (x11**2 + y11**2)**1.5

    
    return (dxdt1, dv_xdt1, dydt1, dv_ydt1,
            dxdt2, dv_xdt2, dydt2, dv_ydt2,
            dxdt3, dv_xdt3, dydt3, dv_ydt3,
            dxdt4, dv_xdt4, dydt4, dv_ydt4,
            dxdt5, dv_xdt5, dydt5, dv_ydt5,
            dxdt6, dv_xdt6, dydt6, dv_ydt6,
            dxdt7, dv_xdt7, dydt7, dv_ydt7,
            dxdt8, dv_xdt8, dydt8, dv_ydt8,
            dxdt9, dv_xdt9, dydt9, dv_ydt9,
            dxdt10, dv_xdt10, dydt10, dv_ydt10,
            dxdt11, dv_xdt11, dydt11, dv_ydt11,
            )

# Определяем начальные значения и параметры
G = 6.67 * 10**(-11)
M = 1.998 * 10**(30)
ae = 149 * 10**9
edge =5 * ae
e = 0.3

def Ve(r, a):
    return np.sqrt((G * M) / a * (2 / (1 - e) - 1 ))

# x0 = 0.5 * ae
# v_x0 = 0
# y0 = 0
# v_y0 = - Ve(0.5*ae, ae)

alpha1 = 70
kappa1 = 0.5
x01 = kappa1 * ae * np.cos(np.deg2rad(alpha1))
v_x01 = - Ve(1, kappa1*ae) * np.sin(np.deg2rad(alpha1))
y01 = kappa1 * ae *  np.sin(np.deg2rad(alpha1))
v_y01 = Ve(1, kappa1*ae) * np.cos(np.deg2rad(alpha1))

alpha2 = 75
kappa2 = 0.501
x02 = kappa2 * ae *  np.cos(np.deg2rad(alpha2))
v_x02 = - Ve(1, kappa2*ae) * np.sin(np.deg2rad(alpha2))
y02 = kappa2 * ae *  np.sin(np.deg2rad(alpha2))
v_y02 = Ve(1, kappa2*ae) * np.cos(np.deg2rad(alpha2))

alpha3 = 80
kappa3 = 0.502
x03 = kappa3 * ae *  np.cos(np.deg2rad(alpha3))
v_x03 = - Ve(1, kappa3*ae) * np.sin(np.deg2rad(alpha3))
y03 = kappa3 * ae *  np.sin(np.deg2rad(alpha3))
v_y03 = Ve(1, kappa3*ae) * np.cos(np.deg2rad(alpha3))

alpha4 = 85
kappa4 = 0.503
x04 = kappa4 * ae * np.cos(np.deg2rad(alpha4))
v_x04 = - Ve(1, kappa4*ae) * np.sin(np.deg2rad(alpha4))
y04 = kappa4 * ae * np.sin(np.deg2rad(alpha4))
v_y04 = Ve(1, kappa4*ae) * np.cos(np.deg2rad(alpha4))

alpha5 = 90
kappa5 = 0.504
x05 = kappa5 * ae * np.cos(np.deg2rad(alpha5))
v_x05 = - Ve(1, kappa5*ae) * np.sin(np.deg2rad(alpha5))
y05 = kappa5 * ae * np.sin(np.deg2rad(alpha5))
v_y05 = Ve(1, kappa5*ae) * np.cos(np.deg2rad(alpha5))

alpha6 = 95
kappa6 = 0.505
x06 = kappa6 * ae * np.cos(np.deg2rad(alpha6))
v_x06 = - Ve(1, kappa6*ae) * np.sin(np.deg2rad(alpha6))
y06 = kappa6 * ae * np.sin(np.deg2rad(alpha6))
v_y06 = Ve(1, kappa6*ae) * np.cos(np.deg2rad(alpha6))

alpha7 = 100
kappa7 = .506
x07 = kappa7 * ae * np.cos(np.deg2rad(alpha7))
v_x07 = - Ve(1, kappa7*ae) * np.sin(np.deg2rad(alpha7))
y07 = kappa7 * ae * np.sin(np.deg2rad(alpha7))
v_y07 = Ve(1, kappa7*ae) * np.cos(np.deg2rad(alpha7))

alpha8 = 105
kappa8 = .507
x08 = kappa8 * ae * np.cos(np.deg2rad(alpha8))
v_x08 = - Ve(1, kappa8*ae) * np.sin(np.deg2rad(alpha8))
y08 = kappa8 * ae * np.sin(np.deg2rad(alpha8))
v_y08 = Ve(1, kappa8*ae) * np.cos(np.deg2rad(alpha8))

alpha9 = 110
kappa9 = .508
x09 = kappa9 * ae * np.cos(np.deg2rad(alpha9))
v_x09 = - Ve(1, kappa9*ae) * np.sin(np.deg2rad(alpha9))
y09 = kappa9 * ae * np.sin(np.deg2rad(alpha9))
v_y09 = Ve(1, kappa9*ae) * np.cos(np.deg2rad(alpha9))

alpha10 = 115
kappa10 = .509
x010 = kappa10 * ae * np.cos(np.deg2rad(alpha10))
v_x010 = - Ve(1, kappa10*ae) * np.sin(np.deg2rad(alpha10))
y010 =  kappa10 * ae * np.sin(np.deg2rad(alpha10))
v_y010 = Ve(1, kappa10*ae) * np.cos(np.deg2rad(alpha10))

alpha11 = 120
kappa11 = .51
x011 = kappa11 * ae * np.cos(np.deg2rad(alpha11))
v_x011 = - Ve(1, kappa11*ae) * np.sin(np.deg2rad(alpha11))
y011 =  kappa11 * ae * np.sin(np.deg2rad(alpha11))
v_y011 = Ve(1, kappa11*ae) * np.cos(np.deg2rad(alpha11))


s0 = (x01, v_x01, y01, v_y01,
      x02, v_x02, y02, v_y02,
      x03, v_x03, y03, v_y03,
      x04, v_x04, y04, v_y04,
      x05, v_x05, y05, v_y05,
      x06, v_x06, y06, v_y06,
      x07, v_x07, y07, v_y07,
      x08, v_x08, y08, v_y08,
      x09, v_x09, y09, v_y09,
      x010, v_x010, y010, v_y010,
      x011, v_x011, y011, v_y011)

sol = odeint(move_func, s0, t)

# Строим решение в виде графика и анимируем
fig, ax = plt.subplots()

number_points = 100
points = []
points_lines = []

for i in range(number_points):
    points.append(plt.plot([], [], 'o', color='r'))
    points_lines.append(plt.plot([], [], '-', color='r'))

def animate(i):
    # for j in range(number_points):
    for j in range(11):
        # Исправление: передаем точки как списки с одним элементом
        points[j][0].set_data([sol[i, 4 * j]], [sol[i, 4 * j + 2]])
        points_lines[j][0].set_data(sol[:i, 4 * j], sol[:i, 4 * j + 2])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

plt.axis('equal')
ax.set_xlim(-edge, edge)
ax.set_ylim(-edge, edge)

plt.plot([0], [0], 'o', color='r', ms=20)

ani.save('galaxy.gif', writer='pillow')
