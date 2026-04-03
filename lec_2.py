import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Определяем переменную величину
frames = 1500
seconds_in_year = 365 * 24 * 60 * 60
years = 10
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
     x11, v_x11, y11, v_y11,
     x12, v_x12, y12, v_y12,
     x13, v_x13, y13, v_y13,
     x14, v_x14, y14, v_y14,
     x15, v_x15, y15, v_y15,
     x16, v_x16, y16, v_y16,
     x17, v_x17, y17, v_y17,
     x18, v_x18, y18, v_y18,
     x19, v_x19, y19, v_y19,
     x20, v_x20, y20, v_y20,
     x21, v_x21, y21, v_y21,
     x22, v_x22, y22, v_y22,
     x0022, v_x0022, y0022, v_y0022) = s
    
    # dxdt = v_x
    # dv_xdt = - G * M * x / (x**2 + y**2)**1.5
    # dydt = v_y
    # dv_ydt = - G * M * y / (x**2 + y**2)**1.5

    dxdt1 = v_x1
    dv_xdt1 = - G * M * x1 / (x1**2 + y1**2)**1.5
    dydt1 = v_y1
    dv_ydt1 = - G * M * y1 / (x1**2 + y1**2)**1.5

    if t > 0.1 * seconds_in_year:
        dxdt2 = v_x2
        dv_xdt2 = - G * M * x2 / (x2**2 + y2**2)**1.5
        dydt2 = v_y2
        dv_ydt2 = - G * M * y2 / (x2**2 + y2**2)**1.5
    else:
        dxdt2 = 0
        dv_xdt2 = 0
        dydt2 = 0
        dv_ydt2 = 0

    if t > 0.2 * seconds_in_year:
        dxdt3 = v_x3
        dv_xdt3 = - G * M * x3 / (x3**2 + y3**2)**1.5
        dydt3 = v_y3
        dv_ydt3 = - G * M * y3 / (x3**2 + y3**2)**1.5
    else:
        dxdt3 = 0
        dv_xdt3 = 0
        dydt3 = 0
        dv_ydt3 = 0

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

    dxdt12 = v_x12
    dv_xdt12 = - G * M * x12 / (x12**2 + y12**2)**1.5
    dydt12 = v_y12
    dv_ydt12 = - G * M * y12 / (x12**2 + y12**2)**1.5

    dxdt13 = v_x13
    dv_xdt13 = - G * M * x13 / (x13**2 + y13**2)**1.5
    dydt13 = v_y13
    dv_ydt13 = - G * M * y13 / (x13**2 + y13**2)**1.5

    dxdt14 = v_x14
    dv_xdt14 = - G * M * x14 / (x14**2 + y14**2)**1.5
    dydt14 = v_y14
    dv_ydt14 = - G * M * y14 / (x14**2 + y14**2)**1.5

    dxdt15 = v_x15
    dv_xdt15 = - G * M * x15 / (x15**2 + y15**2)**1.5
    dydt15 = v_y15
    dv_ydt15 = - G * M * y15 / (x15**2 + y15**2)**1.5

    dxdt16 = v_x16
    dv_xdt16 = - G * M * x16 / (x16**2 + y16**2)**1.5
    dydt16 = v_y16
    dv_ydt16 = - G * M * y16 / (x16**2 + y16**2)**1.5

    dxdt17 = v_x17
    dv_xdt17 = - G * M * x17 / (x17**2 + y17**2)**1.5
    dydt17 = v_y17
    dv_ydt17 = - G * M * y17 / (x17**2 + y17**2)**1.5

    dxdt18 = v_x18
    dv_xdt18 = - G * M * x18 / (x18**2 + y18**2)**1.5
    dydt18 = v_y18
    dv_ydt18 = - G * M * y18 / (x18**2 + y18**2)**1.5

    dxdt19 = v_x19
    dv_xdt19 = - G * M * x19 / (x19**2 + y19**2)**1.5
    dydt19 = v_y19
    dv_ydt19 = - G * M * y19 / (x19**2 + y19**2)**1.5

    dxdt20 = v_x20
    dv_xdt20 = - G * M * x20 / (x20**2 + y20**2)**1.5
    dydt20 = v_y20
    dv_ydt20 = - G * M * y20 / (x20**2 + y20**2)**1.5

    dxdt21 = v_x21
    dv_xdt21 = - G * M * x21 / (x21**2 + y21**2)**1.5
    dydt21 = v_y21
    dv_ydt21 = - G * M * y21 / (x21**2 + y21**2)**1.5

    dxdt22 = v_x22
    dv_xdt22 = - G * M * x22 / (x22**2 + y22**2)**1.5
    dydt22 = v_y22
    dv_ydt22 = - G * M * y22 / (x22**2 + y22**2)**1.5

    
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
            dxdt12, dv_xdt12, dydt12, dv_ydt12,
            dxdt13, dv_xdt13, dydt13, dv_ydt13,
            dxdt14, dv_xdt14, dydt14, dv_ydt14,
            dxdt15, dv_xdt15, dydt15, dv_ydt15,
            dxdt16, dv_xdt16, dydt16, dv_ydt16,
            dxdt17, dv_xdt17, dydt17, dv_ydt17,
            dxdt18, dv_xdt18, dydt18, dv_ydt18,
            dxdt19, dv_xdt19, dydt19, dv_ydt19,
            dxdt20, dv_xdt20, dydt20, dv_ydt20,
            dxdt21, dv_xdt21, dydt21, dv_ydt21,
            dxdt22, dv_xdt22, dydt22, dv_ydt22,
            x0022, v_x0022, y0022, v_y0022)

# Определяем начальные значения и параметры
G = 6.67 * 10**(-11)
M = 1.998 * 10**(30)
ae = 149 * 10**9
edge =5 * ae
e = 0.2

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
kappa2 = 0.55
x02 = kappa2 * ae *  np.cos(np.deg2rad(alpha2))
v_x02 = - Ve(1, kappa2*ae) * np.sin(np.deg2rad(alpha2))
y02 = kappa2 * ae *  np.sin(np.deg2rad(alpha2))
v_y02 = Ve(1, kappa2*ae) * np.cos(np.deg2rad(alpha2))

alpha3 = 80
kappa3 = 0.6
x03 = kappa3 * ae *  np.cos(np.deg2rad(alpha3))
v_x03 = - Ve(1, kappa3*ae) * np.sin(np.deg2rad(alpha3))
y03 = kappa3 * ae *  np.sin(np.deg2rad(alpha3))
v_y03 = Ve(1, kappa3*ae) * np.cos(np.deg2rad(alpha3))

alpha4 = 85
kappa4 = 0.65
x04 = kappa4 * ae * np.cos(np.deg2rad(alpha4))
v_x04 = - Ve(1, kappa4*ae) * np.sin(np.deg2rad(alpha4))
y04 = kappa4 * ae * np.sin(np.deg2rad(alpha4))
v_y04 = Ve(1, kappa4*ae) * np.cos(np.deg2rad(alpha4))

alpha5 = 90
kappa5 = 0.7
x05 = kappa5 * ae * np.cos(np.deg2rad(alpha5))
v_x05 = - Ve(1, kappa5*ae) * np.sin(np.deg2rad(alpha5))
y05 = kappa5 * ae * np.sin(np.deg2rad(alpha5))
v_y05 = Ve(1, kappa5*ae) * np.cos(np.deg2rad(alpha5))

alpha6 = 95
kappa6 = 0.75
x06 = kappa6 * ae * np.cos(np.deg2rad(alpha6))
v_x06 = - Ve(1, kappa6*ae) * np.sin(np.deg2rad(alpha6))
y06 = kappa6 * ae * np.sin(np.deg2rad(alpha6))
v_y06 = Ve(1, kappa6*ae) * np.cos(np.deg2rad(alpha6))

alpha7 = 100
kappa7 = .8
x07 = kappa7 * ae * np.cos(np.deg2rad(alpha7))
v_x07 = - Ve(1, kappa7*ae) * np.sin(np.deg2rad(alpha7))
y07 = kappa7 * ae * np.sin(np.deg2rad(alpha7))
v_y07 = Ve(1, kappa7*ae) * np.cos(np.deg2rad(alpha7))

alpha8 = 105
kappa8 = .85
x08 = kappa8 * ae * np.cos(np.deg2rad(alpha8))
v_x08 = - Ve(1, kappa8*ae) * np.sin(np.deg2rad(alpha8))
y08 = kappa8 * ae * np.sin(np.deg2rad(alpha8))
v_y08 = Ve(1, kappa8*ae) * np.cos(np.deg2rad(alpha8))

alpha9 = 110
kappa9 = .9
x09 = kappa9 * ae * np.cos(np.deg2rad(alpha9))
v_x09 = - Ve(1, kappa9*ae) * np.sin(np.deg2rad(alpha9))
y09 = kappa9 * ae * np.sin(np.deg2rad(alpha9))
v_y09 = Ve(1, kappa9*ae) * np.cos(np.deg2rad(alpha9))

alpha10 = 115
kappa10 = .95
x010 = kappa10 * ae * np.cos(np.deg2rad(alpha10))
v_x010 = - Ve(1, kappa10*ae) * np.sin(np.deg2rad(alpha10))
y010 =  kappa10 * ae * np.sin(np.deg2rad(alpha10))
v_y010 = Ve(1, kappa10*ae) * np.cos(np.deg2rad(alpha10))

alpha11 = 120
kappa11 = 1
x011 = kappa11 * ae * np.cos(np.deg2rad(alpha11))
v_x011 = - Ve(1, kappa11*ae) * np.sin(np.deg2rad(alpha11))
y011 =  kappa11 * ae * np.sin(np.deg2rad(alpha11))
v_y011 = Ve(1, kappa11*ae) * np.cos(np.deg2rad(alpha11))

alpha12 = 250
kappa12 = 0.5
x012 = kappa12 * ae * np.cos(np.deg2rad(alpha12))
v_x012 = - Ve(1, kappa12*ae) * np.sin(np.deg2rad(alpha12))
y012 = kappa12 * ae *  np.sin(np.deg2rad(alpha12))
v_y012 = Ve(1, kappa12*ae) * np.cos(np.deg2rad(alpha12))

alpha13 = 255
kappa13 = 0.55
x013 = kappa13 * ae *  np.cos(np.deg2rad(alpha13))
v_x013 = - Ve(1, kappa13*ae) * np.sin(np.deg2rad(alpha13))
y013 = kappa13 * ae *  np.sin(np.deg2rad(alpha13))
v_y013 = Ve(1, kappa13*ae) * np.cos(np.deg2rad(alpha13))

alpha14 = 260
kappa14 = 0.6
x014 = kappa14 * ae *  np.cos(np.deg2rad(alpha14))
v_x014 = - Ve(1, kappa14*ae) * np.sin(np.deg2rad(alpha14))
y014 = kappa14 * ae *  np.sin(np.deg2rad(alpha14))
v_y014 = Ve(1, kappa14*ae) * np.cos(np.deg2rad(alpha14))

alpha15 = 265
kappa15 = 0.65
x015 = kappa15 * ae * np.cos(np.deg2rad(alpha15))
v_x015 = - Ve(1, kappa15*ae) * np.sin(np.deg2rad(alpha15))
y015 = kappa15 * ae * np.sin(np.deg2rad(alpha15))
v_y015 = Ve(1, kappa15*ae) * np.cos(np.deg2rad(alpha15))

alpha16 = 270
kappa16 = 0.7
x016 = kappa16 * ae * np.cos(np.deg2rad(alpha16))
v_x016 = - Ve(1, kappa16*ae) * np.sin(np.deg2rad(alpha16))
y016 = kappa16 * ae * np.sin(np.deg2rad(alpha16))
v_y016 = Ve(1, kappa16*ae) * np.cos(np.deg2rad(alpha16))

alpha17 = 275
kappa17 = 0.75
x017 = kappa17 * ae * np.cos(np.deg2rad(alpha17))
v_x017 = - Ve(1, kappa17*ae) * np.sin(np.deg2rad(alpha17))
y017 = kappa17 * ae * np.sin(np.deg2rad(alpha17))
v_y017 = Ve(1, kappa17*ae) * np.cos(np.deg2rad(alpha17))

alpha18 = 280
kappa18 = .8
x018 = kappa18 * ae * np.cos(np.deg2rad(alpha18))
v_x018 = - Ve(1, kappa18*ae) * np.sin(np.deg2rad(alpha18))
y018 = kappa18 * ae * np.sin(np.deg2rad(alpha18))
v_y018 = Ve(1, kappa18*ae) * np.cos(np.deg2rad(alpha18))

alpha19 = 285
kappa19 = .85
x019 = kappa19 * ae * np.cos(np.deg2rad(alpha19))
v_x019 = - Ve(1, kappa19*ae) * np.sin(np.deg2rad(alpha19))
y019 = kappa8 * ae * np.sin(np.deg2rad(alpha19))
v_y019 = Ve(1, kappa19*ae) * np.cos(np.deg2rad(alpha19))

alpha20 = 290
kappa20 = .9
x020 = kappa20 * ae * np.cos(np.deg2rad(alpha20))
v_x020 = - Ve(1, kappa20*ae) * np.sin(np.deg2rad(alpha20))
y020 = kappa20 * ae * np.sin(np.deg2rad(alpha20))
v_y020 = Ve(1, kappa20*ae) * np.cos(np.deg2rad(alpha20))

alpha21 = 295
kappa21 = .95
x021 = kappa21 * ae * np.cos(np.deg2rad(alpha21))
v_x021 = - Ve(1, kappa21*ae) * np.sin(np.deg2rad(alpha21))
y021 =  kappa21 * ae * np.sin(np.deg2rad(alpha21))
v_y021 = Ve(1, kappa21*ae) * np.cos(np.deg2rad(alpha21))

alpha22 = 300
kappa22 = 1
x022 = kappa22 * ae * np.cos(np.deg2rad(alpha22))
v_x022 = - Ve(1, kappa22*ae) * np.sin(np.deg2rad(alpha22))
y022 =  kappa22 * ae * np.sin(np.deg2rad(alpha22))
v_y022 = Ve(1, kappa22*ae) * np.cos(np.deg2rad(alpha22))


def Vc(e, a):
    return np.sqrt(G * M / a)

alpha022 = 15
kappa022 = 0.1
x0022 = kappa022 * ae * np.cos(np.deg2rad(alpha022))
v_x0022 = - Vc(0, kappa022*ae) * np.sin(np.deg2rad(alpha022))
y0022 =  kappa022 * ae * np.sin(np.deg2rad(alpha022))
v_y0022 = Vc(0, kappa022*ae) * np.cos(np.deg2rad(alpha022))

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
      x011, v_x011, y011, v_y011,
      x012, v_x012, y012, v_y012,
      x013, v_x013, y013, v_y013,
      x014, v_x014, y014, v_y014,
      x015, v_x015, y015, v_y015,
      x016, v_x016, y016, v_y016,
      x017, v_x017, y017, v_y017,
      x018, v_x018, y018, v_y018,
      x019, v_x019, y019, v_y019,
      x020, v_x020, y020, v_y020,
      x021, v_x021, y021, v_y021,
      x022, v_x022, y022, v_y022,
      x0022, v_x0022, y0022, v_y0022)

sol = odeint(move_func, s0, t)

# Строим решение в виде графика и анимируем
fig, ax = plt.subplots()
ax.set_facecolor('k')

number_points = 100
points = []
points_lines = []

for i in range(number_points):
    # points.append(plt.plot([], [], ',', color='r'))
    points.append(plt.plot([], [], 'o', color='w', ms='0.5'))
    points_lines.append(plt.plot([], [], '-', color='r'))

def animate(i):
    # for j in range(number_points):
    for j in range(23):
        # Исправление: передаем точки как списки с одним элементом
        points[j][0].set_data([sol[i, 4 * j]], [sol[i, 4 * j + 2]])
        # points_lines[j][0].set_data(sol[:i, 4 * j], sol[:i, 4 * j + 2])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

plt.axis('equal')
ax.set_xlim(-edge, edge)
ax.set_ylim(-edge, edge)

# plt.plot([0], [0], 'o', color='w', ms=20)

ani.save('galaxy.gif', writer='pillow')