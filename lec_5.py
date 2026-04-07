import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Определяем переменную величину
frames = 3500
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
     x23, v_x23, y23, v_y23,
     x24, v_x24, y24, v_y24,
     x25, v_x25, y25, v_y25,
     x26, v_x26, y26, v_y26,
     x27, v_x27, y27, v_y27,
     x28, v_x28, y28, v_y28,
     x29, v_x29, y29, v_y29,
     x30, v_x30, y30, v_y30,
     x31, v_x31, y31, v_y31,
     x32, v_x32, y32, v_y32,
     x33, v_x33, y33, v_y33,
     x34, v_x34, y34, v_y34,
     x35, v_x35, y35, v_y35,
     x36, v_x36, y36, v_y36,
     x37, v_x37, y37, v_y37,
     x38, v_x38, y38, v_y38,
     x39, v_x39, y39, v_y39,
     x40, v_x40, y40, v_y40,
     x41, v_x41, y41, v_y41,
     x42, v_x42, y42, v_y42,
     x43, v_x43, y43, v_y43,
     x44, v_x44, y44, v_y44,
     x45, v_x45, y45, v_y45,
     x46, v_x46, y46, v_y46,
     x47, v_x47, y47, v_y47,
     x48, v_x48, y48, v_y48,
     x49, v_x49, y49, v_y49,
     x50, v_x50, y50, v_y50,
     x51, v_x51, y51, v_y51,
     x52, v_x52, y52, v_y52,
     x53, v_x53, y53, v_y53,
     x54, v_x54, y54, v_y54,
     x55, v_x55, y55, v_y55,
     x56, v_x56, y56, v_y56,
     x57, v_x57, y57, v_y57,
     x58, v_x58, y58, v_y58,
     x59, v_x59, y59, v_y59,
     x60, v_x60, y60, v_y60,
     x61, v_x61, y61, v_y61,
     x62, v_x62, y62, v_y62,
     x63, v_x63, y63, v_y63,
     x64, v_x64, y64, v_y64,
     x65, v_x65, y65, v_y65,
     x66, v_x66, y66, v_y66,
     x67, v_x67, y67, v_y67,
     x68, v_x68, y68, v_y68,
     x69, v_x69, y69, v_y69,
     x70, v_x70, y70, v_y70,
     x71, v_x71, y71, v_y71,
     x72, v_x72, y72, v_y72,
     x73, v_x73, y73, v_y73,
     x74, v_x74, y74, v_y74,
     x75, v_x75, y75, v_y75,
     x76, v_x76, y76, v_y76,
     x77, v_x77, y77, v_y77,
     x78, v_x78, y78, v_y78,
     x79, v_x79, y79, v_y79,
     x80, v_x80, y80, v_y80,
     x81, v_x81, y81, v_y81,
     x82, v_x82, y82, v_y82,
     x83, v_x83, y83, v_y83,
     x84, v_x84, y84, v_y84,
     x85, v_x85, y85, v_y85,
     x86, v_x86, y86, v_y86,
     x87, v_x87, y87, v_y87,
     x88, v_x88, y88, v_y88,
     x89, v_x89, y89, v_y89,
     x90, v_x90, y90, v_y90,
     x91, v_x91, y91, v_y91,
     x92, v_x92, y92, v_y92,
     x93, v_x93, y93, v_y93,
     x94, v_x94, y94, v_y94,
     x95, v_x95, y95, v_y95,
     x96, v_x96, y96, v_y96,
     x97, v_x97, y97, v_y97,
     x98, v_x98, y98, v_y98,
     x99, v_x99, y99, v_y99,
     x100, v_x100, y100, v_y100,
     xc1, v_xc1, yc1, v_yc1,
     xc2, v_xc2, yc2, v_yc2,
     xc3, v_xc3, yc3, v_yc3,
     xc4, v_xc4, yc4, v_yc4,
     xc5, v_xc5, yc5, v_yc5,
     xc6, v_xc6, yc6, v_yc6,
     xc7, v_xc7, yc7, v_yc7,
     xc8, v_xc8, yc8, v_yc8,
     xc9, v_xc9, yc9, v_yc9,
     xc10, v_xc10, yc10, v_yc10,
     xc11, v_xc11, yc11, v_yc11,
     xc12, v_xc12, yc12, v_yc12,
     xc13, v_xc13, yc13, v_yc13,
     xc14, v_xc14, yc14, v_yc14,
     xc15, v_xc15, yc15, v_yc15,
     xc16, v_xc16, yc16, v_yc16,
     xc17, v_xc17, yc17, v_yc17,
     xc18, v_xc18, yc18, v_yc18,
     xc19, v_xc19, yc19, v_yc19,
     xc20, v_xc20, yc20, v_yc20,
     xc21, v_xc21, yc21, v_yc21,
     xc22, v_xc22, yc22, v_yc22,
     xc23, v_xc23, yc23, v_yc23,
     xc24, v_xc24, yc24, v_yc24,
     xc25, v_xc25, yc25, v_yc25,
     xc26, v_xc26, yc26, v_yc26,
     xc27, v_xc27, yc27, v_yc27,
     xc28, v_xc28, yc28, v_yc28,
     xc29, v_xc29, yc29, v_yc29,
     xc30, v_xc30, yc30, v_yc30,
     xc31, v_xc31, yc31, v_yc31,
     xc32, v_xc32, yc32, v_yc32,
     xc33, v_xc33, yc33, v_yc33,
     xc34, v_xc34, yc34, v_yc34,
     xc35, v_xc35, yc35, v_yc35) = s

    
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

    if t > 0.3 * seconds_in_year:
        dxdt4 = v_x4
        dv_xdt4 = - G * M * x4 / (x4**2 + y4**2)**1.5
        dydt4 = v_y4
        dv_ydt4 = - G * M * y4 / (x4**2 + y4**2)**1.5
    else:
        dxdt4 = 0
        dv_xdt4 = 0
        dydt4 = 0
        dv_ydt4 = 0

    if t > 0.4 * seconds_in_year:
        dxdt5 = v_x5
        dv_xdt5 = - G * M * x5 / (x5**2 + y5**2)**1.5
        dydt5 = v_y5
        dv_ydt5 = - G * M * y5 / (x5**2 + y5**2)**1.5
    else:
        dxdt5 = 0
        dv_xdt5 = 0
        dydt5 = 0
        dv_ydt5 = 0

    if t > 0.5 * seconds_in_year:
        dxdt6 = v_x6
        dv_xdt6 = - G * M * x6 / (x6**2 + y6**2)**1.5
        dydt6 = v_y6
        dv_ydt6 = - G * M * y6 / (x6**2 + y6**2)**1.5
    else:
        dxdt6 = 0
        dv_xdt6 = 0
        dydt6 = 0
        dv_ydt6 = 0

    if t > 0.6 * seconds_in_year:
        dxdt7 = v_x7
        dv_xdt7 = - G * M * x7 / (x7**2 + y7**2)**1.5
        dydt7 = v_y7
        dv_ydt7 = - G * M * y7 / (x7**2 + y7**2)**1.5
    else:
        dxdt7 = 0
        dv_xdt7 = 0
        dydt7 = 0
        dv_ydt7 = 0

    if t > 0.7 * seconds_in_year:
        dxdt8 = v_x8
        dv_xdt8 = - G * M * x8 / (x8**2 + y8**2)**1.5
        dydt8 = v_y8
        dv_ydt8 = - G * M * y8 / (x8**2 + y8**2)**1.5
    else:
        dxdt8 = 0
        dv_xdt8 = 0
        dydt8 = 0
        dv_ydt8 = 0

    if t > 0.8 * seconds_in_year:
        dxdt9 = v_x9
        dv_xdt9 = - G * M * x9 / (x9**2 + y9**2)**1.5
        dydt9 = v_y9
        dv_ydt9 = - G * M * y9 / (x9**2 + y9**2)**1.5
    else:
        dxdt9 = 0
        dv_xdt9 = 0
        dydt9 = 0
        dv_ydt9 = 0

    if t > 0.9 * seconds_in_year:
        dxdt10 = v_x10
        dv_xdt10 = - G * M * x10 / (x10**2 + y10**2)**1.5
        dydt10 = v_y10
        dv_ydt10 = - G * M * y10 / (x10**2 + y10**2)**1.5
    else:
        dxdt10 = 0
        dv_xdt10 = 0
        dydt10 = 0
        dv_ydt10 = 0

    if t > 1.0 * seconds_in_year:
        dxdt11 = v_x11
        dv_xdt11 = - G * M * x11 / (x11**2 + y11**2)**1.5
        dydt11 = v_y11
        dv_ydt11 = - G * M * y11 / (x11**2 + y11**2)**1.5
    else:
        dxdt11 = 0
        dv_xdt11 = 0
        dydt11 = 0
        dv_ydt11 = 0

    if t > 1.1 * seconds_in_year:
        dxdt12 = v_x12
        dv_xdt12 = - G * M * x12 / (x12**2 + y12**2)**1.5
        dydt12 = v_y12
        dv_ydt12 = - G * M * y12 / (x12**2 + y12**2)**1.5
    else:
        dxdt12 = 0
        dv_xdt12 = 0
        dydt12 = 0
        dv_ydt12 = 0

    if t > 1.2 * seconds_in_year:
        dxdt13 = v_x13
        dv_xdt13 = - G * M * x13 / (x13**2 + y13**2)**1.5
        dydt13 = v_y13
        dv_ydt13 = - G * M * y13 / (x13**2 + y13**2)**1.5
    else:
        dxdt13 = 0
        dv_xdt13 = 0
        dydt13 = 0
        dv_ydt13 = 0

    if t > 1.3 * seconds_in_year:
        dxdt14 = v_x14
        dv_xdt14 = - G * M * x14 / (x14**2 + y14**2)**1.5
        dydt14 = v_y14
        dv_ydt14 = - G * M * y14 / (x14**2 + y14**2)**1.5
    else:
        dxdt14 = 0
        dv_xdt14 = 0
        dydt14 = 0
        dv_ydt14 = 0

    if t > 1.4 * seconds_in_year:
        dxdt15 = v_x15
        dv_xdt15 = - G * M * x15 / (x15**2 + y15**2)**1.5
        dydt15 = v_y15
        dv_ydt15 = - G * M * y15 / (x15**2 + y15**2)**1.5
    else:
        dxdt15 = 0
        dv_xdt15 = 0
        dydt15 = 0
        dv_ydt15 = 0

    if t > 1.5 * seconds_in_year:
        dxdt16 = v_x16
        dv_xdt16 = - G * M * x16 / (x16**2 + y16**2)**1.5
        dydt16 = v_y16
        dv_ydt16 = - G * M * y16 / (x16**2 + y16**2)**1.5
    else:
        dxdt16 = 0
        dv_xdt16 = 0
        dydt16 = 0
        dv_ydt16 = 0

    if t > 1.6 * seconds_in_year:
        dxdt17 = v_x17
        dv_xdt17 = - G * M * x17 / (x17**2 + y17**2)**1.5
        dydt17 = v_y17
        dv_ydt17 = - G * M * y17 / (x17**2 + y17**2)**1.5
    else:
        dxdt17 = 0
        dv_xdt17 = 0
        dydt17 = 0
        dv_ydt17 = 0

    if t > 1.7 * seconds_in_year:
        dxdt18 = v_x18
        dv_xdt18 = - G * M * x18 / (x18**2 + y18**2)**1.5
        dydt18 = v_y18
        dv_ydt18 = - G * M * y18 / (x18**2 + y18**2)**1.5
    else:
        dxdt18 = 0
        dv_xdt18 = 0
        dydt18 = 0
        dv_ydt18 = 0

    if t > 1.8 * seconds_in_year:
        dxdt19 = v_x19
        dv_xdt19 = - G * M * x19 / (x19**2 + y19**2)**1.5
        dydt19 = v_y19
        dv_ydt19 = - G * M * y19 / (x19**2 + y19**2)**1.5
    else:
        dxdt19 = 0
        dv_xdt19 = 0
        dydt19 = 0
        dv_ydt19 = 0

    if t > 1.9 * seconds_in_year:
        dxdt20 = v_x20
        dv_xdt20 = - G * M * x20 / (x20**2 + y20**2)**1.5
        dydt20 = v_y20
        dv_ydt20 = - G * M * y20 / (x20**2 + y20**2)**1.5
    else:
        dxdt20 = 0
        dv_xdt20 = 0
        dydt20 = 0
        dv_ydt20 = 0

    if t > 2.0 * seconds_in_year:
        dxdt21 = v_x21
        dv_xdt21 = - G * M * x21 / (x21**2 + y21**2)**1.5
        dydt21 = v_y21
        dv_ydt21 = - G * M * y21 / (x21**2 + y21**2)**1.5
    else:
        dxdt21 = 0
        dv_xdt21 = 0
        dydt21 = 0
        dv_ydt21 = 0

    if t > 2.1 * seconds_in_year:
        dxdt22 = v_x22
        dv_xdt22 = - G * M * x22 / (x22**2 + y22**2)**1.5
        dydt22 = v_y22
        dv_ydt22 = - G * M * y22 / (x22**2 + y22**2)**1.5
    else:
        dxdt22 = 0
        dv_xdt22 = 0
        dydt22 = 0
        dv_ydt22 = 0

    if t > 2.2 * seconds_in_year:
        dxdt23 = v_x23
        dv_xdt23 = - G * M * x23 / (x23**2 + y23**2)**1.5
        dydt23 = v_y23
        dv_ydt23 = - G * M * y23 / (x23**2 + y23**2)**1.5
    else:
        dxdt23 = 0
        dv_xdt23 = 0
        dydt23 = 0
        dv_ydt23 = 0

    if t > 2.3 * seconds_in_year:
        dxdt24 = v_x24
        dv_xdt24 = - G * M * x24 / (x24**2 + y24**2)**1.5
        dydt24 = v_y24
        dv_ydt24 = - G * M * y24 / (x24**2 + y24**2)**1.5
    else:
        dxdt24 = 0
        dv_xdt24 = 0
        dydt24 = 0
        dv_ydt24 = 0

    if t > 2.4 * seconds_in_year:
        dxdt25 = v_x25
        dv_xdt25 = - G * M * x25 / (x25**2 + y25**2)**1.5
        dydt25 = v_y25
        dv_ydt25 = - G * M * y25 / (x25**2 + y25**2)**1.5
    else:
        dxdt25 = 0
        dv_xdt25 = 0
        dydt25 = 0
        dv_ydt25 = 0

    if t > 2.5 * seconds_in_year:
        dxdt26 = v_x26
        dv_xdt26 = - G * M * x26 / (x26**2 + y26**2)**1.5
        dydt26 = v_y26
        dv_ydt26 = - G * M * y26 / (x26**2 + y26**2)**1.5
    else:
        dxdt26 = 0
        dv_xdt26 = 0
        dydt26 = 0
        dv_ydt26 = 0

    if t > 2.6 * seconds_in_year:
        dxdt27 = v_x27
        dv_xdt27 = - G * M * x27 / (x27**2 + y27**2)**1.5
        dydt27 = v_y27
        dv_ydt27 = - G * M * y27 / (x27**2 + y27**2)**1.5
    else:
        dxdt27 = 0
        dv_xdt27 = 0
        dydt27 = 0
        dv_ydt27 = 0

    if t > 2.7 * seconds_in_year:
        dxdt28 = v_x28
        dv_xdt28 = - G * M * x28 / (x28**2 + y28**2)**1.5
        dydt28 = v_y28
        dv_ydt28 = - G * M * y28 / (x28**2 + y28**2)**1.5
    else:
        dxdt28 = 0
        dv_xdt28 = 0
        dydt28 = 0
        dv_ydt28 = 0

    if t > 2.8 * seconds_in_year:
        dxdt29 = v_x29
        dv_xdt29 = - G * M * x29 / (x29**2 + y29**2)**1.5
        dydt29 = v_y29
        dv_ydt29 = - G * M * y29 / (x29**2 + y29**2)**1.5
    else:
        dxdt29 = 0
        dv_xdt29 = 0
        dydt29 = 0
        dv_ydt29 = 0

    if t > 2.9 * seconds_in_year:
        dxdt30 = v_x30
        dv_xdt30 = - G * M * x30 / (x30**2 + y30**2)**1.5
        dydt30 = v_y30
        dv_ydt30 = - G * M * y30 / (x30**2 + y30**2)**1.5
    else:
        dxdt30 = 0
        dv_xdt30 = 0
        dydt30 = 0
        dv_ydt30 = 0

    if t > 3.0 * seconds_in_year:
        dxdt31 = v_x31
        dv_xdt31 = - G * M * x31 / (x31**2 + y31**2)**1.5
        dydt31 = v_y31
        dv_ydt31 = - G * M * y31 / (x31**2 + y31**2)**1.5
    else:
        dxdt31 = 0
        dv_xdt31 = 0
        dydt31 = 0
        dv_ydt31 = 0

    if t > 3.1 * seconds_in_year:
        dxdt32 = v_x32
        dv_xdt32 = - G * M * x32 / (x32**2 + y32**2)**1.5
        dydt32 = v_y32
        dv_ydt32 = - G * M * y32 / (x32**2 + y32**2)**1.5
    else:
        dxdt32 = 0
        dv_xdt32 = 0
        dydt32 = 0
        dv_ydt32 = 0

    if t > 3.2 * seconds_in_year:
        dxdt33 = v_x33
        dv_xdt33 = - G * M * x33 / (x33**2 + y33**2)**1.5
        dydt33 = v_y33
        dv_ydt33 = - G * M * y33 / (x33**2 + y33**2)**1.5
    else:
        dxdt33 = 0
        dv_xdt33 = 0
        dydt33 = 0
        dv_ydt33 = 0

    if t > 3.3 * seconds_in_year:
        dxdt34 = v_x34
        dv_xdt34 = - G * M * x34 / (x34**2 + y34**2)**1.5
        dydt34 = v_y34
        dv_ydt34 = - G * M * y34 / (x34**2 + y34**2)**1.5
    else:
        dxdt34 = 0
        dv_xdt34 = 0
        dydt34 = 0
        dv_ydt34 = 0

    if t > 3.4 * seconds_in_year:
        dxdt35 = v_x35
        dv_xdt35 = - G * M * x35 / (x35**2 + y35**2)**1.5
        dydt35 = v_y35
        dv_ydt35 = - G * M * y35 / (x35**2 + y35**2)**1.5
    else:
        dxdt35 = 0
        dv_xdt35 = 0
        dydt35 = 0
        dv_ydt35 = 0

    if t > 3.5 * seconds_in_year:
        dxdt36 = v_x36
        dv_xdt36 = - G * M * x36 / (x36**2 + y36**2)**1.5
        dydt36 = v_y36
        dv_ydt36 = - G * M * y36 / (x36**2 + y36**2)**1.5
    else:
        dxdt36 = 0
        dv_xdt36 = 0
        dydt36 = 0
        dv_ydt36 = 0

    if t > 3.6 * seconds_in_year:
        dxdt37 = v_x37
        dv_xdt37 = - G * M * x37 / (x37**2 + y37**2)**1.5
        dydt37 = v_y37
        dv_ydt37 = - G * M * y37 / (x37**2 + y37**2)**1.5
    else:
        dxdt37 = 0
        dv_xdt37 = 0
        dydt37 = 0
        dv_ydt37 = 0

    if t > 3.7 * seconds_in_year:
        dxdt38 = v_x38
        dv_xdt38 = - G * M * x38 / (x38**2 + y38**2)**1.5
        dydt38 = v_y38
        dv_ydt38 = - G * M * y38 / (x38**2 + y38**2)**1.5
    else:
        dxdt38 = 0
        dv_xdt38 = 0
        dydt38 = 0
        dv_ydt38 = 0

    if t > 3.8 * seconds_in_year:
        dxdt39 = v_x39
        dv_xdt39 = - G * M * x39 / (x39**2 + y39**2)**1.5
        dydt39 = v_y39
        dv_ydt39 = - G * M * y39 / (x39**2 + y39**2)**1.5
    else:
        dxdt39 = 0
        dv_xdt39 = 0
        dydt39 = 0
        dv_ydt39 = 0

    if t > 3.9 * seconds_in_year:
        dxdt40 = v_x40
        dv_xdt40 = - G * M * x40 / (x40**2 + y40**2)**1.5
        dydt40 = v_y40
        dv_ydt40 = - G * M * y40 / (x40**2 + y40**2)**1.5
    else:
        dxdt40 = 0
        dv_xdt40 = 0
        dydt40 = 0
        dv_ydt40 = 0

    if t > 4.0 * seconds_in_year:
        dxdt41 = v_x41
        dv_xdt41 = - G * M * x41 / (x41**2 + y41**2)**1.5
        dydt41 = v_y41
        dv_ydt41 = - G * M * y41 / (x41**2 + y41**2)**1.5
    else:
        dxdt41 = 0
        dv_xdt41 = 0
        dydt41 = 0
        dv_ydt41 = 0

    if t > 4.1 * seconds_in_year:
        dxdt42 = v_x42
        dv_xdt42 = - G * M * x42 / (x42**2 + y42**2)**1.5
        dydt42 = v_y42
        dv_ydt42 = - G * M * y42 / (x42**2 + y42**2)**1.5
    else:
        dxdt42 = 0
        dv_xdt42 = 0
        dydt42 = 0
        dv_ydt42 = 0

    if t > 4.2 * seconds_in_year:
        dxdt43 = v_x43
        dv_xdt43 = - G * M * x43 / (x43**2 + y43**2)**1.5
        dydt43 = v_y43
        dv_ydt43 = - G * M * y43 / (x43**2 + y43**2)**1.5
    else:
        dxdt43 = 0
        dv_xdt43 = 0
        dydt43 = 0
        dv_ydt43 = 0

    if t > 4.3 * seconds_in_year:
        dxdt44 = v_x44
        dv_xdt44 = - G * M * x44 / (x44**2 + y44**2)**1.5
        dydt44 = v_y44
        dv_ydt44 = - G * M * y44 / (x44**2 + y44**2)**1.5
    else:
        dxdt44 = 0
        dv_xdt44 = 0
        dydt44 = 0
        dv_ydt44 = 0

    if t > 4.4 * seconds_in_year:
        dxdt45 = v_x45
        dv_xdt45 = - G * M * x45 / (x45**2 + y45**2)**1.5
        dydt45 = v_y45
        dv_ydt45 = - G * M * y45 / (x45**2 + y45**2)**1.5
    else:
        dxdt45 = 0
        dv_xdt45 = 0
        dydt45 = 0
        dv_ydt45 = 0

    if t > 4.5 * seconds_in_year:
        dxdt46 = v_x46
        dv_xdt46 = - G * M * x46 / (x46**2 + y46**2)**1.5
        dydt46 = v_y46
        dv_ydt46 = - G * M * y46 / (x46**2 + y46**2)**1.5
    else:
        dxdt46 = 0
        dv_xdt46 = 0
        dydt46 = 0
        dv_ydt46 = 0

    if t > 4.6 * seconds_in_year:
        dxdt47 = v_x47
        dv_xdt47 = - G * M * x47 / (x47**2 + y47**2)**1.5
        dydt47 = v_y47
        dv_ydt47 = - G * M * y47 / (x47**2 + y47**2)**1.5
    else:
        dxdt47 = 0
        dv_xdt47 = 0
        dydt47 = 0
        dv_ydt47 = 0

    if t > 4.7 * seconds_in_year:
        dxdt48 = v_x48
        dv_xdt48 = - G * M * x48 / (x48**2 + y48**2)**1.5
        dydt48 = v_y48
        dv_ydt48 = - G * M * y48 / (x48**2 + y48**2)**1.5
    else:
        dxdt48 = 0
        dv_xdt48 = 0
        dydt48 = 0
        dv_ydt48 = 0

    if t > 4.8 * seconds_in_year:
        dxdt49 = v_x49
        dv_xdt49 = - G * M * x49 / (x49**2 + y49**2)**1.5
        dydt49 = v_y49
        dv_ydt49 = - G * M * y49 / (x49**2 + y49**2)**1.5
    else:
        dxdt49 = 0
        dv_xdt49 = 0
        dydt49 = 0
        dv_ydt49 = 0

    if t > 4.9 * seconds_in_year:
        dxdt50 = v_x50
        dv_xdt50 = - G * M * x50 / (x50**2 + y50**2)**1.5
        dydt50 = v_y50
        dv_ydt50 = - G * M * y50 / (x50**2 + y50**2)**1.5
    else:
        dxdt50 = 0
        dv_xdt50 = 0
        dydt50 = 0
        dv_ydt50 = 0

    if t > 0 * seconds_in_year:
        dxdt51 = v_x51
        dv_xdt51 = - G * M * x51 / (x51**2 + y51**2)**1.5
        dydt51 = v_y51
        dv_ydt51 = - G * M * y51 / (x51**2 + y51**2)**1.5
    else:
        dxdt51 = 0
        dv_xdt51 = 0
        dydt51 = 0
        dv_ydt51 = 0

    if t > 0.1 * seconds_in_year:
        dxdt52 = v_x52
        dv_xdt52 = - G * M * x52 / (x52**2 + y52**2)**1.5
        dydt52 = v_y52
        dv_ydt52 = - G * M * y52 / (x52**2 + y52**2)**1.5
    else:
        dxdt52 = 0
        dv_xdt52 = 0
        dydt52 = 0
        dv_ydt52 = 0

    if t > 0.2 * seconds_in_year:
        dxdt53 = v_x53
        dv_xdt53 = - G * M * x53 / (x53**2 + y53**2)**1.5
        dydt53 = v_y53
        dv_ydt53 = - G * M * y53 / (x53**2 + y53**2)**1.5
    else:
        dxdt53 = 0
        dv_xdt53 = 0
        dydt53 = 0
        dv_ydt53 = 0

    if t > 0.3 * seconds_in_year:
        dxdt54 = v_x54
        dv_xdt54 = - G * M * x54 / (x54**2 + y54**2)**1.5
        dydt54 = v_y54
        dv_ydt54 = - G * M * y54 / (x54**2 + y54**2)**1.5
    else:
        dxdt54 = 0
        dv_xdt54 = 0
        dydt54 = 0
        dv_ydt54 = 0

    if t > 0.4 * seconds_in_year:
        dxdt55 = v_x55
        dv_xdt55 = - G * M * x55 / (x55**2 + y55**2)**1.5
        dydt55 = v_y55
        dv_ydt55 = - G * M * y55 / (x55**2 + y55**2)**1.5
    else:
        dxdt55 = 0
        dv_xdt55 = 0
        dydt55 = 0
        dv_ydt55 = 0

    if t > 0.5 * seconds_in_year:
        dxdt56 = v_x56
        dv_xdt56 = - G * M * x56 / (x56**2 + y56**2)**1.5
        dydt56 = v_y56
        dv_ydt56 = - G * M * y56 / (x56**2 + y56**2)**1.5
    else:
        dxdt56 = 0
        dv_xdt56 = 0
        dydt56 = 0
        dv_ydt56 = 0

    if t > 0.6 * seconds_in_year:
        dxdt57 = v_x57
        dv_xdt57 = - G * M * x57 / (x57**2 + y57**2)**1.5
        dydt57 = v_y57
        dv_ydt57 = - G * M * y57 / (x57**2 + y57**2)**1.5
    else:
        dxdt57 = 0
        dv_xdt57 = 0
        dydt57 = 0
        dv_ydt57 = 0

    if t > 0.7 * seconds_in_year:
        dxdt58 = v_x58
        dv_xdt58 = - G * M * x58 / (x58**2 + y58**2)**1.5
        dydt58 = v_y58
        dv_ydt58 = - G * M * y58 / (x58**2 + y58**2)**1.5
    else:
        dxdt58 = 0
        dv_xdt58 = 0
        dydt58 = 0
        dv_ydt58 = 0

    if t > 0.8 * seconds_in_year:
        dxdt59 = v_x59
        dv_xdt59 = - G * M * x59 / (x59**2 + y59**2)**1.5
        dydt59 = v_y59
        dv_ydt59 = - G * M * y59 / (x59**2 + y59**2)**1.5
    else:
        dxdt59 = 0
        dv_xdt59 = 0
        dydt59 = 0
        dv_ydt59 = 0

    if t > 0.9 * seconds_in_year:
        dxdt60 = v_x60
        dv_xdt60 = - G * M * x60 / (x60**2 + y60**2)**1.5
        dydt60 = v_y60
        dv_ydt60 = - G * M * y60 / (x60**2 + y60**2)**1.5
    else:
        dxdt60 = 0
        dv_xdt60 = 0
        dydt60 = 0
        dv_ydt60 = 0

    if t > 1 * seconds_in_year:
        dxdt61 = v_x61
        dv_xdt61 = - G * M * x61 / (x61**2 + y61**2)**1.5
        dydt61 = v_y61
        dv_ydt61 = - G * M * y61 / (x61**2 + y61**2)**1.5
    else:
        dxdt61 = 0
        dv_xdt61 = 0
        dydt61 = 0
        dv_ydt61 = 0

    if t > 1.1 * seconds_in_year:
        dxdt62 = v_x62
        dv_xdt62 = - G * M * x62 / (x62**2 + y62**2)**1.5
        dydt62 = v_y62
        dv_ydt62 = - G * M * y62 / (x62**2 + y62**2)**1.5
    else:
        dxdt62 = 0
        dv_xdt62 = 0
        dydt62 = 0
        dv_ydt62 = 0

    if t > 1.2 * seconds_in_year:
        dxdt63 = v_x63
        dv_xdt63 = - G * M * x63 / (x63**2 + y63**2)**1.5
        dydt63 = v_y63
        dv_ydt63 = - G * M * y63 / (x63**2 + y63**2)**1.5
    else:
        dxdt63 = 0
        dv_xdt63 = 0
        dydt63 = 0
        dv_ydt63 = 0

    if t > 1.3 * seconds_in_year:
        dxdt64 = v_x64
        dv_xdt64 = - G * M * x64 / (x64**2 + y64**2)**1.5
        dydt64 = v_y64
        dv_ydt64 = - G * M * y64 / (x64**2 + y64**2)**1.5
    else:
        dxdt64 = 0
        dv_xdt64 = 0
        dydt64 = 0
        dv_ydt64 = 0

    if t > 1.4 * seconds_in_year:
        dxdt65 = v_x65
        dv_xdt65 = - G * M * x65 / (x65**2 + y65**2)**1.5
        dydt65 = v_y65
        dv_ydt65 = - G * M * y65 / (x65**2 + y65**2)**1.5
    else:
        dxdt65 = 0
        dv_xdt65 = 0
        dydt65 = 0
        dv_ydt65 = 0

    if t > 1.5 * seconds_in_year:
        dxdt66 = v_x66
        dv_xdt66 = - G * M * x66 / (x66**2 + y66**2)**1.5
        dydt66 = v_y66
        dv_ydt66 = - G * M * y66 / (x66**2 + y66**2)**1.5
    else:
        dxdt66 = 0
        dv_xdt66 = 0
        dydt66 = 0
        dv_ydt66 = 0

    if t > 1.6 * seconds_in_year:
        dxdt67 = v_x67
        dv_xdt67 = - G * M * x67 / (x67**2 + y67**2)**1.5
        dydt67 = v_y67
        dv_ydt67 = - G * M * y67 / (x67**2 + y67**2)**1.5
    else:
        dxdt67 = 0
        dv_xdt67 = 0
        dydt67 = 0
        dv_ydt67 = 0

    if t > 1.7 * seconds_in_year:
        dxdt68 = v_x68
        dv_xdt68 = - G * M * x68 / (x68**2 + y68**2)**1.5
        dydt68 = v_y68
        dv_ydt68 = - G * M * y68 / (x68**2 + y68**2)**1.5
    else:
        dxdt68 = 0
        dv_xdt68 = 0
        dydt68 = 0
        dv_ydt68 = 0

    if t > 1.8 * seconds_in_year:
        dxdt69 = v_x69
        dv_xdt69 = - G * M * x69 / (x69**2 + y69**2)**1.5
        dydt69 = v_y69
        dv_ydt69 = - G * M * y69 / (x69**2 + y69**2)**1.5
    else:
        dxdt69 = 0
        dv_xdt69 = 0
        dydt69 = 0
        dv_ydt69 = 0

    if t > 1.9 * seconds_in_year:
        dxdt70 = v_x70
        dv_xdt70 = - G * M * x70 / (x70**2 + y70**2)**1.5
        dydt70 = v_y70
        dv_ydt70 = - G * M * y70 / (x70**2 + y70**2)**1.5
    else:
        dxdt70 = 0
        dv_xdt70 = 0
        dydt70 = 0
        dv_ydt70 = 0

    if t > 2 * seconds_in_year:
        dxdt71 = v_x71
        dv_xdt71 = - G * M * x71 / (x71**2 + y71**2)**1.5
        dydt71 = v_y71
        dv_ydt71 = - G * M * y71 / (x71**2 + y71**2)**1.5
    else:
        dxdt71 = 0
        dv_xdt71 = 0
        dydt71 = 0
        dv_ydt71 = 0

    if t > 2.1 * seconds_in_year:
        dxdt72 = v_x72
        dv_xdt72 = - G * M * x72 / (x72**2 + y72**2)**1.5
        dydt72 = v_y72
        dv_ydt72 = - G * M * y72 / (x72**2 + y72**2)**1.5
    else:
        dxdt72 = 0
        dv_xdt72 = 0
        dydt72 = 0
        dv_ydt72 = 0

    if t > 2.2 * seconds_in_year:
        dxdt73 = v_x73
        dv_xdt73 = - G * M * x73 / (x73**2 + y73**2)**1.5
        dydt73 = v_y73
        dv_ydt73 = - G * M * y73 / (x73**2 + y73**2)**1.5
    else:
        dxdt73 = 0
        dv_xdt73 = 0
        dydt73 = 0
        dv_ydt73 = 0

    if t > 2.3 * seconds_in_year:
        dxdt74 = v_x74
        dv_xdt74 = - G * M * x74 / (x74**2 + y74**2)**1.5
        dydt74 = v_y74
        dv_ydt74 = - G * M * y74 / (x74**2 + y74**2)**1.5
    else:
        dxdt74 = 0
        dv_xdt74 = 0
        dydt74 = 0
        dv_ydt74 = 0

    if t > 2.4 * seconds_in_year:
        dxdt75 = v_x75
        dv_xdt75 = - G * M * x75 / (x75**2 + y75**2)**1.5
        dydt75 = v_y75
        dv_ydt75 = - G * M * y75 / (x75**2 + y75**2)**1.5
    else:
        dxdt75 = 0
        dv_xdt75 = 0
        dydt75 = 0
        dv_ydt75 = 0

    if t > 2.5 * seconds_in_year:
        dxdt76 = v_x76
        dv_xdt76 = - G * M * x76 / (x76**2 + y76**2)**1.5
        dydt76 = v_y76
        dv_ydt76 = - G * M * y76 / (x76**2 + y76**2)**1.5
    else:
        dxdt76 = 0
        dv_xdt76 = 0
        dydt76 = 0
        dv_ydt76 = 0

    if t > 2.6 * seconds_in_year:
        dxdt77 = v_x77
        dv_xdt77 = - G * M * x77 / (x77**2 + y77**2)**1.5
        dydt77 = v_y77
        dv_ydt77 = - G * M * y77 / (x77**2 + y77**2)**1.5
    else:
        dxdt77 = 0
        dv_xdt77 = 0
        dydt77 = 0
        dv_ydt77 = 0

    if t > 2.7 * seconds_in_year:
        dxdt78 = v_x78
        dv_xdt78 = - G * M * x78 / (x78**2 + y78**2)**1.5
        dydt78 = v_y78
        dv_ydt78 = - G * M * y78 / (x78**2 + y78**2)**1.5
    else:
        dxdt78 = 0
        dv_xdt78 = 0
        dydt78 = 0
        dv_ydt78 = 0

    if t > 2.8 * seconds_in_year:
        dxdt79 = v_x79
        dv_xdt79 = - G * M * x79 / (x79**2 + y79**2)**1.5
        dydt79 = v_y79
        dv_ydt79 = - G * M * y79 / (x79**2 + y79**2)**1.5
    else:
        dxdt79 = 0
        dv_xdt79 = 0
        dydt79 = 0
        dv_ydt79 = 0

    if t > 2.9 * seconds_in_year:
        dxdt80 = v_x80
        dv_xdt80 = - G * M * x80 / (x80**2 + y80**2)**1.5
        dydt80 = v_y80
        dv_ydt80 = - G * M * y80 / (x80**2 + y80**2)**1.5
    else:
        dxdt80 = 0
        dv_xdt80 = 0
        dydt80 = 0
        dv_ydt80 = 0

    if t > 3 * seconds_in_year:
        dxdt81 = v_x81
        dv_xdt81 = - G * M * x81 / (x81**2 + y81**2)**1.5
        dydt81 = v_y81
        dv_ydt81 = - G * M * y81 / (x81**2 + y81**2)**1.5
    else:
        dxdt81 = 0
        dv_xdt81 = 0
        dydt81 = 0
        dv_ydt81 = 0

    if t > 3.1 * seconds_in_year:
        dxdt82 = v_x82
        dv_xdt82 = - G * M * x82 / (x82**2 + y82**2)**1.5
        dydt82 = v_y82
        dv_ydt82 = - G * M * y82 / (x82**2 + y82**2)**1.5
    else:
        dxdt82 = 0
        dv_xdt82 = 0
        dydt82 = 0
        dv_ydt82 = 0

    if t > 3.2 * seconds_in_year:
        dxdt83 = v_x83
        dv_xdt83 = - G * M * x83 / (x83**2 + y83**2)**1.5
        dydt83 = v_y83
        dv_ydt83 = - G * M * y83 / (x83**2 + y83**2)**1.5
    else:
        dxdt83 = 0
        dv_xdt83 = 0
        dydt83 = 0
        dv_ydt83 = 0

    if t > 3.3 * seconds_in_year:
        dxdt84 = v_x84
        dv_xdt84 = - G * M * x84 / (x84**2 + y84**2)**1.5
        dydt84 = v_y84
        dv_ydt84 = - G * M * y84 / (x84**2 + y84**2)**1.5
    else:
        dxdt84 = 0
        dv_xdt84 = 0
        dydt84 = 0
        dv_ydt84 = 0

    if t > 3.4 * seconds_in_year:
        dxdt85 = v_x85
        dv_xdt85 = - G * M * x85 / (x85**2 + y85**2)**1.5
        dydt85 = v_y85
        dv_ydt85 = - G * M * y85 / (x85**2 + y85**2)**1.5
    else:
        dxdt85 = 0
        dv_xdt85 = 0
        dydt85 = 0
        dv_ydt85 = 0

    if t > 3.5 * seconds_in_year:
        dxdt86 = v_x86
        dv_xdt86 = - G * M * x86 / (x86**2 + y86**2)**1.5
        dydt86 = v_y86
        dv_ydt86 = - G * M * y86 / (x86**2 + y86**2)**1.5
    else:
        dxdt86 = 0
        dv_xdt86 = 0
        dydt86 = 0
        dv_ydt86 = 0

    if t > 3.6 * seconds_in_year:
        dxdt87 = v_x87
        dv_xdt87 = - G * M * x87 / (x87**2 + y87**2)**1.5
        dydt87 = v_y87
        dv_ydt87 = - G * M * y87 / (x87**2 + y87**2)**1.5
    else:
        dxdt87 = 0
        dv_xdt87 = 0
        dydt87 = 0
        dv_ydt87 = 0

    if t > 3.7 * seconds_in_year:
        dxdt88 = v_x88
        dv_xdt88 = - G * M * x88 / (x88**2 + y88**2)**1.5
        dydt88 = v_y88
        dv_ydt88 = - G * M * y88 / (x88**2 + y88**2)**1.5
    else:
        dxdt88 = 0
        dv_xdt88 = 0
        dydt88 = 0
        dv_ydt88 = 0

    if t > 3.8 * seconds_in_year:
        dxdt89 = v_x89
        dv_xdt89 = - G * M * x89 / (x89**2 + y89**2)**1.5
        dydt89 = v_y89
        dv_ydt89 = - G * M * y89 / (x89**2 + y89**2)**1.5
    else:
        dxdt89 = 0
        dv_xdt89 = 0
        dydt89 = 0
        dv_ydt89 = 0

    if t > 3.9 * seconds_in_year:
        dxdt90 = v_x90
        dv_xdt90 = - G * M * x90 / (x90**2 + y90**2)**1.5
        dydt90 = v_y90
        dv_ydt90 = - G * M * y90 / (x90**2 + y90**2)**1.5
    else:
        dxdt90 = 0
        dv_xdt90 = 0
        dydt90 = 0
        dv_ydt90 = 0

    if t > 4 * seconds_in_year:
        dxdt91 = v_x91
        dv_xdt91 = - G * M * x91 / (x91**2 + y91**2)**1.5
        dydt91 = v_y91
        dv_ydt91 = - G * M * y91 / (x91**2 + y91**2)**1.5
    else:
        dxdt91 = 0
        dv_xdt91 = 0
        dydt91 = 0
        dv_ydt91 = 0

    if t > 4.1 * seconds_in_year:
        dxdt92 = v_x92
        dv_xdt92 = - G * M * x92 / (x92**2 + y92**2)**1.5
        dydt92 = v_y92
        dv_ydt92 = - G * M * y92 / (x92**2 + y92**2)**1.5
    else:
        dxdt92 = 0
        dv_xdt92 = 0
        dydt92 = 0
        dv_ydt92 = 0

    if t > 4.2 * seconds_in_year:
        dxdt93 = v_x93
        dv_xdt93 = - G * M * x93 / (x93**2 + y93**2)**1.5
        dydt93 = v_y93
        dv_ydt93 = - G * M * y93 / (x93**2 + y93**2)**1.5
    else:
        dxdt93 = 0
        dv_xdt93 = 0
        dydt93 = 0
        dv_ydt93 = 0

    if t > 4.3 * seconds_in_year:
        dxdt94 = v_x94
        dv_xdt94 = - G * M * x94 / (x94**2 + y94**2)**1.5
        dydt94 = v_y94
        dv_ydt94 = - G * M * y94 / (x94**2 + y94**2)**1.5
    else:
        dxdt94 = 0
        dv_xdt94 = 0
        dydt94 = 0
        dv_ydt94 = 0

    if t > 4.4 * seconds_in_year:
        dxdt95 = v_x95
        dv_xdt95 = - G * M * x95 / (x95**2 + y95**2)**1.5
        dydt95 = v_y95
        dv_ydt95 = - G * M * y95 / (x95**2 + y95**2)**1.5
    else:
        dxdt95 = 0
        dv_xdt95 = 0
        dydt95 = 0
        dv_ydt95 = 0

    if t > 4.5 * seconds_in_year:
        dxdt96 = v_x96
        dv_xdt96 = - G * M * x96 / (x96**2 + y96**2)**1.5
        dydt96 = v_y96
        dv_ydt96 = - G * M * y96 / (x96**2 + y96**2)**1.5
    else:
        dxdt96 = 0
        dv_xdt96 = 0
        dydt96 = 0
        dv_ydt96 = 0

    if t > 4.6 * seconds_in_year:
        dxdt97 = v_x97
        dv_xdt97 = - G * M * x97 / (x97**2 + y97**2)**1.5
        dydt97 = v_y97
        dv_ydt97 = - G * M * y97 / (x97**2 + y97**2)**1.5
    else:
        dxdt97 = 0
        dv_xdt97 = 0
        dydt97 = 0
        dv_ydt97 = 0

    if t > 4.7 * seconds_in_year:
        dxdt98 = v_x98
        dv_xdt98 = - G * M * x98 / (x98**2 + y98**2)**1.5
        dydt98 = v_y98
        dv_ydt98 = - G * M * y98 / (x98**2 + y98**2)**1.5
    else:
        dxdt98 = 0
        dv_xdt98 = 0
        dydt98 = 0
        dv_ydt98 = 0

    if t > 4.8 * seconds_in_year:
        dxdt99 = v_x99
        dv_xdt99 = - G * M * x99 / (x99**2 + y99**2)**1.5
        dydt99 = v_y99
        dv_ydt99 = - G * M * y99 / (x99**2 + y99**2)**1.5
    else:
        dxdt99 = 0
        dv_xdt99 = 0
        dydt99 = 0
        dv_ydt99 = 0

    if t > 4.9 * seconds_in_year:
        dxdt100 = v_x100
        dv_xdt100 = - G * M * x100 / (x100**2 + y100**2)**1.5
        dydt100 = v_y100
        dv_ydt100 = - G * M * y100 / (x100**2 + y100**2)**1.5
    else:
        dxdt100 = 0
        dv_xdt100 = 0
        dydt100 = 0
        dv_ydt100 = 0
        
    
    #35 центр точек
    # 35 точек круга (1-я сразу, остальные с шагом 0.1 года)
    dxdtc1 = v_xc1
    dv_xdtc1 = - G * M * xc1 / (xc1**2 + yc1**2)**1.5
    dydtc1 = v_yc1
    dv_ydtc1 = - G * M * yc1 / (xc1**2 + yc1**2)**1.5

    if t > 0.1 * seconds_in_year:
        dxdtc2 = v_xc2
        dv_xdtc2 = - G * M * xc2 / (xc2**2 + yc2**2)**1.5
        dydtc2 = v_yc2
        dv_ydtc2 = - G * M * yc2 / (xc2**2 + yc2**2)**1.5
    else:
        dxdtc2 = 0
        dv_xdtc2 = 0
        dydtc2 = 0
        dv_ydtc2 = 0

    if t > 0.2 * seconds_in_year:
        dxdtc3 = v_xc3
        dv_xdtc3 = - G * M * xc3 / (xc3**2 + yc3**2)**1.5
        dydtc3 = v_yc3
        dv_ydtc3 = - G * M * yc3 / (xc3**2 + yc3**2)**1.5
    else:
        dxdtc3 = 0
        dv_xdtc3 = 0
        dydtc3 = 0
        dv_ydtc3 = 0

    if t > 0.3 * seconds_in_year:
        dxdtc4 = v_xc4
        dv_xdtc4 = - G * M * xc4 / (xc4**2 + yc4**2)**1.5
        dydtc4 = v_yc4
        dv_ydtc4 = - G * M * yc4 / (xc4**2 + yc4**2)**1.5
    else:
        dxdtc4 = 0
        dv_xdtc4 = 0
        dydtc4 = 0
        dv_ydtc4 = 0

    if t > 0.4 * seconds_in_year:
        dxdtc5 = v_xc5
        dv_xdtc5 = - G * M * xc5 / (xc5**2 + yc5**2)**1.5
        dydtc5 = v_yc5
        dv_ydtc5 = - G * M * yc5 / (xc5**2 + yc5**2)**1.5
    else:
        dxdtc5 = 0
        dv_xdtc5 = 0
        dydtc5 = 0
        dv_ydtc5 = 0

    if t > 0.5 * seconds_in_year:
        dxdtc6 = v_xc6
        dv_xdtc6 = - G * M * xc6 / (xc6**2 + yc6**2)**1.5
        dydtc6 = v_yc6
        dv_ydtc6 = - G * M * yc6 / (xc6**2 + yc6**2)**1.5
    else:
        dxdtc6 = 0
        dv_xdtc6 = 0
        dydtc6 = 0
        dv_ydtc6 = 0

    if t > 0.6 * seconds_in_year:
        dxdtc7 = v_xc7
        dv_xdtc7 = - G * M * xc7 / (xc7**2 + yc7**2)**1.5
        dydtc7 = v_yc7
        dv_ydtc7 = - G * M * yc7 / (xc7**2 + yc7**2)**1.5
    else:
        dxdtc7 = 0
        dv_xdtc7 = 0
        dydtc7 = 0
        dv_ydtc7 = 0

    if t > 0.7 * seconds_in_year:
        dxdtc8 = v_xc8
        dv_xdtc8 = - G * M * xc8 / (xc8**2 + yc8**2)**1.5
        dydtc8 = v_yc8
        dv_ydtc8 = - G * M * yc8 / (xc8**2 + yc8**2)**1.5
    else:
        dxdtc8 = 0
        dv_xdtc8 = 0
        dydtc8 = 0
        dv_ydtc8 = 0

    if t > 0.8 * seconds_in_year:
        dxdtc9 = v_xc9
        dv_xdtc9 = - G * M * xc9 / (xc9**2 + yc9**2)**1.5
        dydtc9 = v_yc9
        dv_ydtc9 = - G * M * yc9 / (xc9**2 + yc9**2)**1.5
    else:
        dxdtc9 = 0
        dv_xdtc9 = 0
        dydtc9 = 0
        dv_ydtc9 = 0

    if t > 0.9 * seconds_in_year:
        dxdtc10 = v_xc10
        dv_xdtc10 = - G * M * xc10 / (xc10**2 + yc10**2)**1.5
        dydtc10 = v_yc10
        dv_ydtc10 = - G * M * yc10 / (xc10**2 + yc10**2)**1.5
    else:
        dxdtc10 = 0
        dv_xdtc10 = 0
        dydtc10 = 0
        dv_ydtc10 = 0

    if t > 1.0 * seconds_in_year:
        dxdtc11 = v_xc11
        dv_xdtc11 = - G * M * xc11 / (xc11**2 + yc11**2)**1.5
        dydtc11 = v_yc11
        dv_ydtc11 = - G * M * yc11 / (xc11**2 + yc11**2)**1.5
    else:
        dxdtc11 = 0
        dv_xdtc11 = 0
        dydtc11 = 0
        dv_ydtc11 = 0

    if t > 1.1 * seconds_in_year:
        dxdtc12 = v_xc12
        dv_xdtc12 = - G * M * xc12 / (xc12**2 + yc12**2)**1.5
        dydtc12 = v_yc12
        dv_ydtc12 = - G * M * yc12 / (xc12**2 + yc12**2)**1.5
    else:
        dxdtc12 = 0
        dv_xdtc12 = 0
        dydtc12 = 0
        dv_ydtc12 = 0

    if t > 1.2 * seconds_in_year:
        dxdtc13 = v_xc13
        dv_xdtc13 = - G * M * xc13 / (xc13**2 + yc13**2)**1.5
        dydtc13 = v_yc13
        dv_ydtc13 = - G * M * yc13 / (xc13**2 + yc13**2)**1.5
    else:
        dxdtc13 = 0
        dv_xdtc13 = 0
        dydtc13 = 0
        dv_ydtc13 = 0

    if t > 1.3 * seconds_in_year:
        dxdtc14 = v_xc14
        dv_xdtc14 = - G * M * xc14 / (xc14**2 + yc14**2)**1.5
        dydtc14 = v_yc14
        dv_ydtc14 = - G * M * yc14 / (xc14**2 + yc14**2)**1.5
    else:
        dxdtc14 = 0
        dv_xdtc14 = 0
        dydtc14 = 0
        dv_ydtc14 = 0

    if t > 1.4 * seconds_in_year:
        dxdtc15 = v_xc15
        dv_xdtc15 = - G * M * xc15 / (xc15**2 + yc15**2)**1.5
        dydtc15 = v_yc15
        dv_ydtc15 = - G * M * yc15 / (xc15**2 + yc15**2)**1.5
    else:
        dxdtc15 = 0
        dv_xdtc15 = 0
        dydtc15 = 0
        dv_ydtc15 = 0

    if t > 1.5 * seconds_in_year:
        dxdtc16 = v_xc16
        dv_xdtc16 = - G * M * xc16 / (xc16**2 + yc16**2)**1.5
        dydtc16 = v_yc16
        dv_ydtc16 = - G * M * yc16 / (xc16**2 + yc16**2)**1.5
    else:
        dxdtc16 = 0
        dv_xdtc16 = 0
        dydtc16 = 0
        dv_ydtc16 = 0

    if t > 1.6 * seconds_in_year:
        dxdtc17 = v_xc17
        dv_xdtc17 = - G * M * xc17 / (xc17**2 + yc17**2)**1.5
        dydtc17 = v_yc17
        dv_ydtc17 = - G * M * yc17 / (xc17**2 + yc17**2)**1.5
    else:
        dxdtc17 = 0
        dv_xdtc17 = 0
        dydtc17 = 0
        dv_ydtc17 = 0

    if t > 1.7 * seconds_in_year:
        dxdtc18 = v_xc18
        dv_xdtc18 = - G * M * xc18 / (xc18**2 + yc18**2)**1.5
        dydtc18 = v_yc18
        dv_ydtc18 = - G * M * yc18 / (xc18**2 + yc18**2)**1.5
    else:
        dxdtc18 = 0
        dv_xdtc18 = 0
        dydtc18 = 0
        dv_ydtc18 = 0

    if t > 1.8 * seconds_in_year:
        dxdtc19 = v_xc19
        dv_xdtc19 = - G * M * xc19 / (xc19**2 + yc19**2)**1.5
        dydtc19 = v_yc19
        dv_ydtc19 = - G * M * yc19 / (xc19**2 + yc19**2)**1.5
    else:
        dxdtc19 = 0
        dv_xdtc19 = 0
        dydtc19 = 0
        dv_ydtc19 = 0

    if t > 1.9 * seconds_in_year:
        dxdtc20 = v_xc20
        dv_xdtc20 = - G * M * xc20 / (xc20**2 + yc20**2)**1.5
        dydtc20 = v_yc20
        dv_ydtc20 = - G * M * yc20 / (xc20**2 + yc20**2)**1.5
    else:
        dxdtc20 = 0
        dv_xdtc20 = 0
        dydtc20 = 0
        dv_ydtc20 = 0

    if t > 2 * seconds_in_year:
        dxdtc21 = v_xc21
        dv_xdtc21 = - G * M * xc21 / (xc21**2 + yc21**2)**1.5
        dydtc21 = v_yc21
        dv_ydtc21 = - G * M * yc21 / (xc21**2 + yc21**2)**1.5
    else:
        dxdtc21 = 0
        dv_xdtc21 = 0
        dydtc21 = 0
        dv_ydtc21 = 0

    if t > 2.1 * seconds_in_year:
        dxdtc22 = v_xc22
        dv_xdtc22 = - G * M * xc22 / (xc22**2 + yc22**2)**1.5
        dydtc22 = v_yc22
        dv_ydtc22 = - G * M * yc22 / (xc22**2 + yc22**2)**1.5
    else:
        dxdtc22 = 0
        dv_xdtc22 = 0
        dydtc22 = 0
        dv_ydtc22 = 0

    if t > 2.2 * seconds_in_year:
        dxdtc23 = v_xc23
        dv_xdtc23 = - G * M * xc23 / (xc23**2 + yc23**2)**1.5
        dydtc23 = v_yc23
        dv_ydtc23 = - G * M * yc23 / (xc23**2 + yc23**2)**1.5
    else:
        dxdtc23 = 0
        dv_xdtc23 = 0
        dydtc23 = 0
        dv_ydtc23 = 0

    if t > 2.3 * seconds_in_year:
        dxdtc24 = v_xc24
        dv_xdtc24 = - G * M * xc24 / (xc24**2 + yc24**2)**1.5
        dydtc24 = v_yc24
        dv_ydtc24 = - G * M * yc24 / (xc24**2 + yc24**2)**1.5
    else:
        dxdtc24 = 0
        dv_xdtc24 = 0
        dydtc24 = 0
        dv_ydtc24 = 0

    if t > 2.4 * seconds_in_year:
        dxdtc25 = v_xc25
        dv_xdtc25 = - G * M * xc25 / (xc25**2 + yc25**2)**1.5
        dydtc25 = v_yc25
        dv_ydtc25 = - G * M * yc25 / (xc25**2 + yc25**2)**1.5
    else:
        dxdtc25 = 0
        dv_xdtc25 = 0
        dydtc25 = 0
        dv_ydtc25 = 0

    if t > 2.5 * seconds_in_year:
        dxdtc26 = v_xc26
        dv_xdtc26 = - G * M * xc26 / (xc26**2 + yc26**2)**1.5
        dydtc26 = v_yc26
        dv_ydtc26 = - G * M * yc26 / (xc26**2 + yc26**2)**1.5
    else:
        dxdtc26 = 0
        dv_xdtc26 = 0
        dydtc26 = 0
        dv_ydtc26 = 0

    if t > 2.6 * seconds_in_year:
        dxdtc27 = v_xc27
        dv_xdtc27 = - G * M * xc27 / (xc27**2 + yc27**2)**1.5
        dydtc27 = v_yc27
        dv_ydtc27 = - G * M * yc27 / (xc27**2 + yc27**2)**1.5
    else:
        dxdtc27 = 0
        dv_xdtc27 = 0
        dydtc27 = 0
        dv_ydtc27 = 0

    if t > 2.7 * seconds_in_year:
        dxdtc28 = v_xc28
        dv_xdtc28 = - G * M * xc28 / (xc28**2 + yc28**2)**1.5
        dydtc28 = v_yc28
        dv_ydtc28 = - G * M * yc28 / (xc28**2 + yc28**2)**1.5
    else:
        dxdtc28 = 0
        dv_xdtc28 = 0
        dydtc28 = 0
        dv_ydtc28 = 0

    if t > 2.8 * seconds_in_year:
        dxdtc29 = v_xc29
        dv_xdtc29 = - G * M * xc29 / (xc29**2 + yc29**2)**1.5
        dydtc29 = v_yc29
        dv_ydtc29 = - G * M * yc29 / (xc29**2 + yc29**2)**1.5
    else:
        dxdtc29 = 0
        dv_xdtc29 = 0
        dydtc29 = 0
        dv_ydtc29 = 0

    if t > 2.9 * seconds_in_year:
        dxdtc30 = v_xc30
        dv_xdtc30 = - G * M * xc30 / (xc30**2 + yc30**2)**1.5
        dydtc30 = v_yc30
        dv_ydtc30 = - G * M * yc30 / (xc30**2 + yc30**2)**1.5
    else:
        dxdtc30 = 0
        dv_xdtc30 = 0
        dydtc30 = 0
        dv_ydtc30 = 0

    if t > 3 * seconds_in_year:
        dxdtc31 = v_xc31
        dv_xdtc31 = - G * M * xc31 / (xc31**2 + yc31**2)**1.5
        dydtc31 = v_yc31
        dv_ydtc31 = - G * M * yc31 / (xc31**2 + yc31**2)**1.5
    else:
        dxdtc31 = 0
        dv_xdtc31 = 0
        dydtc31 = 0
        dv_ydtc31 = 0

    if t > 3.1 * seconds_in_year:
        dxdtc32 = v_xc32
        dv_xdtc32 = - G * M * xc32 / (xc32**2 + yc32**2)**1.5
        dydtc32 = v_yc32
        dv_ydtc32 = - G * M * yc32 / (xc32**2 + yc32**2)**1.5
    else:
        dxdtc32 = 0
        dv_xdtc32 = 0
        dydtc32 = 0
        dv_ydtc32 = 0

    if t > 3.2 * seconds_in_year:
        dxdtc33 = v_xc33
        dv_xdtc33 = - G * M * xc33 / (xc33**2 + yc33**2)**1.5
        dydtc33 = v_yc33
        dv_ydtc33 = - G * M * yc33 / (xc33**2 + yc33**2)**1.5
    else:
        dxdtc33 = 0
        dv_xdtc33 = 0
        dydtc33 = 0
        dv_ydtc33 = 0

    if t > 3.3 * seconds_in_year:
        dxdtc34 = v_xc34
        dv_xdtc34 = - G * M * xc34 / (xc34**2 + yc34**2)**1.5
        dydtc34 = v_yc34
        dv_ydtc34 = - G * M * yc34 / (xc34**2 + yc34**2)**1.5
    else:
        dxdtc34 = 0
        dv_xdtc34 = 0
        dydtc34 = 0
        dv_ydtc34 = 0

    if t > 3.4 * seconds_in_year:
        dxdtc35 = v_xc35
        dv_xdtc35 = - G * M * xc35 / (xc35**2 + yc35**2)**1.5
        dydtc35 = v_yc35
        dv_ydtc35 = - G * M * yc35 / (xc35**2 + yc35**2)**1.5
    else:
        dxdtc35 = 0
        dv_xdtc35 = 0
        dydtc35 = 0
        dv_ydtc35 = 0

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
        dxdt23, dv_xdt23, dydt23, dv_ydt23,
        dxdt24, dv_xdt24, dydt24, dv_ydt24,
        dxdt25, dv_xdt25, dydt25, dv_ydt25,
        dxdt26, dv_xdt26, dydt26, dv_ydt26,
        dxdt27, dv_xdt27, dydt27, dv_ydt27,
        dxdt28, dv_xdt28, dydt28, dv_ydt28,
        dxdt29, dv_xdt29, dydt29, dv_ydt29,
        dxdt30, dv_xdt30, dydt30, dv_ydt30,
        dxdt31, dv_xdt31, dydt31, dv_ydt31,
        dxdt32, dv_xdt32, dydt32, dv_ydt32,
        dxdt33, dv_xdt33, dydt33, dv_ydt33,
        dxdt34, dv_xdt34, dydt34, dv_ydt34,
        dxdt35, dv_xdt35, dydt35, dv_ydt35,
        dxdt36, dv_xdt36, dydt36, dv_ydt36,
        dxdt37, dv_xdt37, dydt37, dv_ydt37,
        dxdt38, dv_xdt38, dydt38, dv_ydt38,
        dxdt39, dv_xdt39, dydt39, dv_ydt39,
        dxdt40, dv_xdt40, dydt40, dv_ydt40,
        dxdt41, dv_xdt41, dydt41, dv_ydt41,
        dxdt42, dv_xdt42, dydt42, dv_ydt42,
        dxdt43, dv_xdt43, dydt43, dv_ydt43,
        dxdt44, dv_xdt44, dydt44, dv_ydt44,
        dxdt45, dv_xdt45, dydt45, dv_ydt45,
        dxdt46, dv_xdt46, dydt46, dv_ydt46,
        dxdt47, dv_xdt47, dydt47, dv_ydt47,
        dxdt48, dv_xdt48, dydt48, dv_ydt48,
        dxdt49, dv_xdt49, dydt49, dv_ydt49,
        dxdt50, dv_xdt50, dydt50, dv_ydt50,
        dxdt51, dv_xdt51, dydt51, dv_ydt51,
        dxdt52, dv_xdt52, dydt52, dv_ydt52,
        dxdt53, dv_xdt53, dydt53, dv_ydt53,
        dxdt54, dv_xdt54, dydt54, dv_ydt54,
        dxdt55, dv_xdt55, dydt55, dv_ydt55,
        dxdt56, dv_xdt56, dydt56, dv_ydt56,
        dxdt57, dv_xdt57, dydt57, dv_ydt57,
        dxdt58, dv_xdt58, dydt58, dv_ydt58,
        dxdt59, dv_xdt59, dydt59, dv_ydt59,
        dxdt60, dv_xdt60, dydt60, dv_ydt60,
        dxdt61, dv_xdt61, dydt61, dv_ydt61,
        dxdt62, dv_xdt62, dydt62, dv_ydt62,
        dxdt63, dv_xdt63, dydt63, dv_ydt63,
        dxdt64, dv_xdt64, dydt64, dv_ydt64,
        dxdt65, dv_xdt65, dydt65, dv_ydt65,
        dxdt66, dv_xdt66, dydt66, dv_ydt66,
        dxdt67, dv_xdt67, dydt67, dv_ydt67,
        dxdt68, dv_xdt68, dydt68, dv_ydt68,
        dxdt69, dv_xdt69, dydt69, dv_ydt69,
        dxdt70, dv_xdt70, dydt70, dv_ydt70,
        dxdt71, dv_xdt71, dydt71, dv_ydt71,
        dxdt72, dv_xdt72, dydt72, dv_ydt72,
        dxdt73, dv_xdt73, dydt73, dv_ydt73,
        dxdt74, dv_xdt74, dydt74, dv_ydt74,
        dxdt75, dv_xdt75, dydt75, dv_ydt75,
        dxdt76, dv_xdt76, dydt76, dv_ydt76,
        dxdt77, dv_xdt77, dydt77, dv_ydt77,
        dxdt78, dv_xdt78, dydt78, dv_ydt78,
        dxdt79, dv_xdt79, dydt79, dv_ydt79,
        dxdt80, dv_xdt80, dydt80, dv_ydt80,
        dxdt81, dv_xdt81, dydt81, dv_ydt81,
        dxdt82, dv_xdt82, dydt82, dv_ydt82,
        dxdt83, dv_xdt83, dydt83, dv_ydt83,
        dxdt84, dv_xdt84, dydt84, dv_ydt84,
        dxdt85, dv_xdt85, dydt85, dv_ydt85,
        dxdt86, dv_xdt86, dydt86, dv_ydt86,
        dxdt87, dv_xdt87, dydt87, dv_ydt87,
        dxdt88, dv_xdt88, dydt88, dv_ydt88,
        dxdt89, dv_xdt89, dydt89, dv_ydt89,
        dxdt90, dv_xdt90, dydt90, dv_ydt90,
        dxdt91, dv_xdt91, dydt91, dv_ydt91,
        dxdt92, dv_xdt92, dydt92, dv_ydt92,
        dxdt93, dv_xdt93, dydt93, dv_ydt93,
        dxdt94, dv_xdt94, dydt94, dv_ydt94,
        dxdt95, dv_xdt95, dydt95, dv_ydt95,
        dxdt96, dv_xdt96, dydt96, dv_ydt96,
        dxdt97, dv_xdt97, dydt97, dv_ydt97,
        dxdt98, dv_xdt98, dydt98, dv_ydt98,
        dxdt99, dv_xdt99, dydt99, dv_ydt99,
        dxdt100, dv_xdt100, dydt100, dv_ydt100,
        dxdtc1, dv_xdtc1, dydtc1, dv_ydtc1,
        dxdtc2, dv_xdtc2, dydtc2, dv_ydtc2,
        dxdtc3, dv_xdtc3, dydtc3, dv_ydtc3,
        dxdtc4, dv_xdtc4, dydtc4, dv_ydtc4,
        dxdtc5, dv_xdtc5, dydtc5, dv_ydtc5,
        dxdtc6, dv_xdtc6, dydtc6, dv_ydtc6,
        dxdtc7, dv_xdtc7, dydtc7, dv_ydtc7,
        dxdtc8, dv_xdtc8, dydtc8, dv_ydtc8,
        dxdtc9, dv_xdtc9, dydtc9, dv_ydtc9,
        dxdtc10, dv_xdtc10, dydtc10, dv_ydtc10,
        dxdtc11, dv_xdtc11, dydtc11, dv_ydtc11,
        dxdtc12, dv_xdtc12, dydtc12, dv_ydtc12,
        dxdtc13, dv_xdtc13, dydtc13, dv_ydtc13,
        dxdtc14, dv_xdtc14, dydtc14, dv_ydtc14,
        dxdtc15, dv_xdtc15, dydtc15, dv_ydtc15,
        dxdtc16, dv_xdtc16, dydtc16, dv_ydtc16,
        dxdtc17, dv_xdtc17, dydtc17, dv_ydtc17,
        dxdtc18, dv_xdtc18, dydtc18, dv_ydtc18,
        dxdtc19, dv_xdtc19, dydtc19, dv_ydtc19,
        dxdtc20, dv_xdtc20, dydtc20, dv_ydtc20,
        dxdtc21, dv_xdtc21, dydtc21, dv_ydtc21,
        dxdtc22, dv_xdtc22, dydtc22, dv_ydtc22,
        dxdtc23, dv_xdtc23, dydtc23, dv_ydtc23,
        dxdtc24, dv_xdtc24, dydtc24, dv_ydtc24,
        dxdtc25, dv_xdtc25, dydtc25, dv_ydtc25,
        dxdtc26, dv_xdtc26, dydtc26, dv_ydtc26,
        dxdtc27, dv_xdtc27, dydtc27, dv_ydtc27,
        dxdtc28, dv_xdtc28, dydtc28, dv_ydtc28,
        dxdtc29, dv_xdtc29, dydtc29, dv_ydtc29,
        dxdtc30, dv_xdtc30, dydtc30, dv_ydtc30,
        dxdtc31, dv_xdtc31, dydtc31, dv_ydtc31,
        dxdtc32, dv_xdtc32, dydtc32, dv_ydtc32,
        dxdtc33, dv_xdtc33, dydtc33, dv_ydtc33,
        dxdtc34, dv_xdtc34, dydtc34, dv_ydtc34,
        dxdtc35, dv_xdtc35, dydtc35, dv_ydtc35)

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


alpha1 = 70.0
kappa1 = 0.5
x01 = kappa1 * ae * np.cos(np.deg2rad(alpha1))
v_x01 = - Ve(1, kappa1*ae) * np.sin(np.deg2rad(alpha1))
y01 = kappa1 * ae * np.sin(np.deg2rad(alpha1))
v_y01 = Ve(1, kappa1*ae) * np.cos(np.deg2rad(alpha1))

alpha2 = 71.02
kappa2 = 0.51
x02 = kappa2 * ae * np.cos(np.deg2rad(alpha2))
v_x02 = - Ve(1, kappa2*ae) * np.sin(np.deg2rad(alpha2))
y02 = kappa2 * ae * np.sin(np.deg2rad(alpha2))
v_y02 = Ve(1, kappa2*ae) * np.cos(np.deg2rad(alpha2))

alpha3 = 72.04
kappa3 = 0.52
x03 = kappa3 * ae * np.cos(np.deg2rad(alpha3))
v_x03 = - Ve(1, kappa3*ae) * np.sin(np.deg2rad(alpha3))
y03 = kappa3 * ae * np.sin(np.deg2rad(alpha3))
v_y03 = Ve(1, kappa3*ae) * np.cos(np.deg2rad(alpha3))

alpha4 = 73.06
kappa4 = 0.53
x04 = kappa4 * ae * np.cos(np.deg2rad(alpha4))
v_x04 = - Ve(1, kappa4*ae) * np.sin(np.deg2rad(alpha4))
y04 = kappa4 * ae * np.sin(np.deg2rad(alpha4))
v_y04 = Ve(1, kappa4*ae) * np.cos(np.deg2rad(alpha4))

alpha5 = 74.08
kappa5 = 0.54
x05 = kappa5 * ae * np.cos(np.deg2rad(alpha5))
v_x05 = - Ve(1, kappa5*ae) * np.sin(np.deg2rad(alpha5))
y05 = kappa5 * ae * np.sin(np.deg2rad(alpha5))
v_y05 = Ve(1, kappa5*ae) * np.cos(np.deg2rad(alpha5))

alpha6 = 75.1
kappa6 = 0.55
x06 = kappa6 * ae * np.cos(np.deg2rad(alpha6))
v_x06 = - Ve(1, kappa6*ae) * np.sin(np.deg2rad(alpha6))
y06 = kappa6 * ae * np.sin(np.deg2rad(alpha6))
v_y06 = Ve(1, kappa6*ae) * np.cos(np.deg2rad(alpha6))

alpha7 = 76.12
kappa7 = 0.56
x07 = kappa7 * ae * np.cos(np.deg2rad(alpha7))
v_x07 = - Ve(1, kappa7*ae) * np.sin(np.deg2rad(alpha7))
y07 = kappa7 * ae * np.sin(np.deg2rad(alpha7))
v_y07 = Ve(1, kappa7*ae) * np.cos(np.deg2rad(alpha7))

alpha8 = 77.14
kappa8 = 0.57
x08 = kappa8 * ae * np.cos(np.deg2rad(alpha8))
v_x08 = - Ve(1, kappa8*ae) * np.sin(np.deg2rad(alpha8))
y08 = kappa8 * ae * np.sin(np.deg2rad(alpha8))
v_y08 = Ve(1, kappa8*ae) * np.cos(np.deg2rad(alpha8))

alpha9 = 78.16
kappa9 = 0.58
x09 = kappa9 * ae * np.cos(np.deg2rad(alpha9))
v_x09 = - Ve(1, kappa9*ae) * np.sin(np.deg2rad(alpha9))
y09 = kappa9 * ae * np.sin(np.deg2rad(alpha9))
v_y09 = Ve(1, kappa9*ae) * np.cos(np.deg2rad(alpha9))

alpha10 = 79.18
kappa10 = 0.59
x010 = kappa10 * ae * np.cos(np.deg2rad(alpha10))
v_x010 = - Ve(1, kappa10*ae) * np.sin(np.deg2rad(alpha10))
y010 = kappa10 * ae * np.sin(np.deg2rad(alpha10))
v_y010 = Ve(1, kappa10*ae) * np.cos(np.deg2rad(alpha10))

alpha11 = 80.2
kappa11 = 0.6
x011 = kappa11 * ae * np.cos(np.deg2rad(alpha11))
v_x011 = - Ve(1, kappa11*ae) * np.sin(np.deg2rad(alpha11))
y011 = kappa11 * ae * np.sin(np.deg2rad(alpha11))
v_y011 = Ve(1, kappa11*ae) * np.cos(np.deg2rad(alpha11))

alpha12 = 81.22
kappa12 = 0.61
x012 = kappa12 * ae * np.cos(np.deg2rad(alpha12))
v_x012 = - Ve(1, kappa12*ae) * np.sin(np.deg2rad(alpha12))
y012 = kappa12 * ae * np.sin(np.deg2rad(alpha12))
v_y012 = Ve(1, kappa12*ae) * np.cos(np.deg2rad(alpha12))

alpha13 = 82.24
kappa13 = 0.62
x013 = kappa13 * ae * np.cos(np.deg2rad(alpha13))
v_x013 = - Ve(1, kappa13*ae) * np.sin(np.deg2rad(alpha13))
y013 = kappa13 * ae * np.sin(np.deg2rad(alpha13))
v_y013 = Ve(1, kappa13*ae) * np.cos(np.deg2rad(alpha13))

alpha14 = 83.27
kappa14 = 0.63
x014 = kappa14 * ae * np.cos(np.deg2rad(alpha14))
v_x014 = - Ve(1, kappa14*ae) * np.sin(np.deg2rad(alpha14))
y014 = kappa14 * ae * np.sin(np.deg2rad(alpha14))
v_y014 = Ve(1, kappa14*ae) * np.cos(np.deg2rad(alpha14))

alpha15 = 84.29
kappa15 = 0.64
x015 = kappa15 * ae * np.cos(np.deg2rad(alpha15))
v_x015 = - Ve(1, kappa15*ae) * np.sin(np.deg2rad(alpha15))
y015 = kappa15 * ae * np.sin(np.deg2rad(alpha15))
v_y015 = Ve(1, kappa15*ae) * np.cos(np.deg2rad(alpha15))

alpha16 = 85.31
kappa16 = 0.65
x016 = kappa16 * ae * np.cos(np.deg2rad(alpha16))
v_x016 = - Ve(1, kappa16*ae) * np.sin(np.deg2rad(alpha16))
y016 = kappa16 * ae * np.sin(np.deg2rad(alpha16))
v_y016 = Ve(1, kappa16*ae) * np.cos(np.deg2rad(alpha16))

alpha17 = 86.33
kappa17 = 0.66
x017 = kappa17 * ae * np.cos(np.deg2rad(alpha17))
v_x017 = - Ve(1, kappa17*ae) * np.sin(np.deg2rad(alpha17))
y017 = kappa17 * ae * np.sin(np.deg2rad(alpha17))
v_y017 = Ve(1, kappa17*ae) * np.cos(np.deg2rad(alpha17))

alpha18 = 87.35
kappa18 = 0.67
x018 = kappa18 * ae * np.cos(np.deg2rad(alpha18))
v_x018 = - Ve(1, kappa18*ae) * np.sin(np.deg2rad(alpha18))
y018 = kappa18 * ae * np.sin(np.deg2rad(alpha18))
v_y018 = Ve(1, kappa18*ae) * np.cos(np.deg2rad(alpha18))

alpha19 = 88.37
kappa19 = 0.68
x019 = kappa19 * ae * np.cos(np.deg2rad(alpha19))
v_x019 = - Ve(1, kappa19*ae) * np.sin(np.deg2rad(alpha19))
y019 = kappa19 * ae * np.sin(np.deg2rad(alpha19))
v_y019 = Ve(1, kappa19*ae) * np.cos(np.deg2rad(alpha19))

alpha20 = 89.39
kappa20 = 0.69
x020 = kappa20 * ae * np.cos(np.deg2rad(alpha20))
v_x020 = - Ve(1, kappa20*ae) * np.sin(np.deg2rad(alpha20))
y020 = kappa20 * ae * np.sin(np.deg2rad(alpha20))
v_y020 = Ve(1, kappa20*ae) * np.cos(np.deg2rad(alpha20))

alpha21 = 90.41
kappa21 = 0.7
x021 = kappa21 * ae * np.cos(np.deg2rad(alpha21))
v_x021 = - Ve(1, kappa21*ae) * np.sin(np.deg2rad(alpha21))
y021 = kappa21 * ae * np.sin(np.deg2rad(alpha21))
v_y021 = Ve(1, kappa21*ae) * np.cos(np.deg2rad(alpha21))

alpha22 = 91.43
kappa22 = 0.71
x022 = kappa22 * ae * np.cos(np.deg2rad(alpha22))
v_x022 = - Ve(1, kappa22*ae) * np.sin(np.deg2rad(alpha22))
y022 = kappa22 * ae * np.sin(np.deg2rad(alpha22))
v_y022 = Ve(1, kappa22*ae) * np.cos(np.deg2rad(alpha22))

alpha23 = 92.45
kappa23 = 0.72
x023 = kappa23 * ae * np.cos(np.deg2rad(alpha23))
v_x023 = - Ve(1, kappa23*ae) * np.sin(np.deg2rad(alpha23))
y023 = kappa23 * ae * np.sin(np.deg2rad(alpha23))
v_y023 = Ve(1, kappa23*ae) * np.cos(np.deg2rad(alpha23))

alpha24 = 93.47
kappa24 = 0.73
x024 = kappa24 * ae * np.cos(np.deg2rad(alpha24))
v_x024 = - Ve(1, kappa24*ae) * np.sin(np.deg2rad(alpha24))
y024 = kappa24 * ae * np.sin(np.deg2rad(alpha24))
v_y024 = Ve(1, kappa24*ae) * np.cos(np.deg2rad(alpha24))

alpha25 = 94.49
kappa25 = 0.74
x025 = kappa25 * ae * np.cos(np.deg2rad(alpha25))
v_x025 = - Ve(1, kappa25*ae) * np.sin(np.deg2rad(alpha25))
y025 = kappa25 * ae * np.sin(np.deg2rad(alpha25))
v_y025 = Ve(1, kappa25*ae) * np.cos(np.deg2rad(alpha25))

alpha26 = 95.51
kappa26 = 0.75
x026 = kappa26 * ae * np.cos(np.deg2rad(alpha26))
v_x026 = - Ve(1, kappa26*ae) * np.sin(np.deg2rad(alpha26))
y026 = kappa26 * ae * np.sin(np.deg2rad(alpha26))
v_y026 = Ve(1, kappa26*ae) * np.cos(np.deg2rad(alpha26))

alpha27 = 96.53
kappa27 = 0.76
x027 = kappa27 * ae * np.cos(np.deg2rad(alpha27))
v_x027 = - Ve(1, kappa27*ae) * np.sin(np.deg2rad(alpha27))
y027 = kappa27 * ae * np.sin(np.deg2rad(alpha27))
v_y027 = Ve(1, kappa27*ae) * np.cos(np.deg2rad(alpha27))

alpha28 = 97.55
kappa28 = 0.77
x028 = kappa28 * ae * np.cos(np.deg2rad(alpha28))
v_x028 = - Ve(1, kappa28*ae) * np.sin(np.deg2rad(alpha28))
y028 = kappa28 * ae * np.sin(np.deg2rad(alpha28))
v_y028 = Ve(1, kappa28*ae) * np.cos(np.deg2rad(alpha28))

alpha29 = 98.57
kappa29 = 0.78
x029 = kappa29 * ae * np.cos(np.deg2rad(alpha29))
v_x029 = - Ve(1, kappa29*ae) * np.sin(np.deg2rad(alpha29))
y029 = kappa29 * ae * np.sin(np.deg2rad(alpha29))
v_y029 = Ve(1, kappa29*ae) * np.cos(np.deg2rad(alpha29))

alpha30 = 99.59
kappa30 = 0.79
x030 = kappa30 * ae * np.cos(np.deg2rad(alpha30))
v_x030 = - Ve(1, kappa30*ae) * np.sin(np.deg2rad(alpha30))
y030 = kappa30 * ae * np.sin(np.deg2rad(alpha30))
v_y030 = Ve(1, kappa30*ae) * np.cos(np.deg2rad(alpha30))

alpha31 = 100.61
kappa31 = 0.8
x031 = kappa31 * ae * np.cos(np.deg2rad(alpha31))
v_x031 = - Ve(1, kappa31*ae) * np.sin(np.deg2rad(alpha31))
y031 = kappa31 * ae * np.sin(np.deg2rad(alpha31))
v_y031 = Ve(1, kappa31*ae) * np.cos(np.deg2rad(alpha31))

alpha32 = 101.63
kappa32 = 0.81
x032 = kappa32 * ae * np.cos(np.deg2rad(alpha32))
v_x032 = - Ve(1, kappa32*ae) * np.sin(np.deg2rad(alpha32))
y032 = kappa32 * ae * np.sin(np.deg2rad(alpha32))
v_y032 = Ve(1, kappa32*ae) * np.cos(np.deg2rad(alpha32))

alpha33 = 102.65
kappa33 = 0.82
x033 = kappa33 * ae * np.cos(np.deg2rad(alpha33))
v_x033 = - Ve(1, kappa33*ae) * np.sin(np.deg2rad(alpha33))
y033 = kappa33 * ae * np.sin(np.deg2rad(alpha33))
v_y033 = Ve(1, kappa33*ae) * np.cos(np.deg2rad(alpha33))

alpha34 = 103.67
kappa34 = 0.83
x034 = kappa34 * ae * np.cos(np.deg2rad(alpha34))
v_x034 = - Ve(1, kappa34*ae) * np.sin(np.deg2rad(alpha34))
y034 = kappa34 * ae * np.sin(np.deg2rad(alpha34))
v_y034 = Ve(1, kappa34*ae) * np.cos(np.deg2rad(alpha34))

alpha35 = 104.69
kappa35 = 0.84
x035 = kappa35 * ae * np.cos(np.deg2rad(alpha35))
v_x035 = - Ve(1, kappa35*ae) * np.sin(np.deg2rad(alpha35))
y035 = kappa35 * ae * np.sin(np.deg2rad(alpha35))
v_y035 = Ve(1, kappa35*ae) * np.cos(np.deg2rad(alpha35))

alpha36 = 105.71
kappa36 = 0.85
x036 = kappa36 * ae * np.cos(np.deg2rad(alpha36))
v_x036 = - Ve(1, kappa36*ae) * np.sin(np.deg2rad(alpha36))
y036 = kappa36 * ae * np.sin(np.deg2rad(alpha36))
v_y036 = Ve(1, kappa36*ae) * np.cos(np.deg2rad(alpha36))

alpha37 = 106.73
kappa37 = 0.86
x037 = kappa37 * ae * np.cos(np.deg2rad(alpha37))
v_x037 = - Ve(1, kappa37*ae) * np.sin(np.deg2rad(alpha37))
y037 = kappa37 * ae * np.sin(np.deg2rad(alpha37))
v_y037 = Ve(1, kappa37*ae) * np.cos(np.deg2rad(alpha37))

alpha38 = 107.76
kappa38 = 0.87
x038 = kappa38 * ae * np.cos(np.deg2rad(alpha38))
v_x038 = - Ve(1, kappa38*ae) * np.sin(np.deg2rad(alpha38))
y038 = kappa38 * ae * np.sin(np.deg2rad(alpha38))
v_y038 = Ve(1, kappa38*ae) * np.cos(np.deg2rad(alpha38))

alpha39 = 108.78
kappa39 = 0.88
x039 = kappa39 * ae * np.cos(np.deg2rad(alpha39))
v_x039 = - Ve(1, kappa39*ae) * np.sin(np.deg2rad(alpha39))
y039 = kappa39 * ae * np.sin(np.deg2rad(alpha39))
v_y039 = Ve(1, kappa39*ae) * np.cos(np.deg2rad(alpha39))

alpha40 = 109.8
kappa40 = 0.89
x040 = kappa40 * ae * np.cos(np.deg2rad(alpha40))
v_x040 = - Ve(1, kappa40*ae) * np.sin(np.deg2rad(alpha40))
y040 = kappa40 * ae * np.sin(np.deg2rad(alpha40))
v_y040 = Ve(1, kappa40*ae) * np.cos(np.deg2rad(alpha40))

alpha41 = 110.82
kappa41 = 0.9
x041 = kappa41 * ae * np.cos(np.deg2rad(alpha41))
v_x041 = - Ve(1, kappa41*ae) * np.sin(np.deg2rad(alpha41))
y041 = kappa41 * ae * np.sin(np.deg2rad(alpha41))
v_y041 = Ve(1, kappa41*ae) * np.cos(np.deg2rad(alpha41))

alpha42 = 111.84
kappa42 = 0.91
x042 = kappa42 * ae * np.cos(np.deg2rad(alpha42))
v_x042 = - Ve(1, kappa42*ae) * np.sin(np.deg2rad(alpha42))
y042 = kappa42 * ae * np.sin(np.deg2rad(alpha42))
v_y042 = Ve(1, kappa42*ae) * np.cos(np.deg2rad(alpha42))

alpha43 = 112.86
kappa43 = 0.92
x043 = kappa43 * ae * np.cos(np.deg2rad(alpha43))
v_x043 = - Ve(1, kappa43*ae) * np.sin(np.deg2rad(alpha43))
y043 = kappa43 * ae * np.sin(np.deg2rad(alpha43))
v_y043 = Ve(1, kappa43*ae) * np.cos(np.deg2rad(alpha43))

alpha44 = 113.88
kappa44 = 0.93
x044 = kappa44 * ae * np.cos(np.deg2rad(alpha44))
v_x044 = - Ve(1, kappa44*ae) * np.sin(np.deg2rad(alpha44))
y044 = kappa44 * ae * np.sin(np.deg2rad(alpha44))
v_y044 = Ve(1, kappa44*ae) * np.cos(np.deg2rad(alpha44))

alpha45 = 114.9
kappa45 = 0.94
x045 = kappa45 * ae * np.cos(np.deg2rad(alpha45))
v_x045 = - Ve(1, kappa45*ae) * np.sin(np.deg2rad(alpha45))
y045 = kappa45 * ae * np.sin(np.deg2rad(alpha45))
v_y045 = Ve(1, kappa45*ae) * np.cos(np.deg2rad(alpha45))

alpha46 = 115.92
kappa46 = 0.95
x046 = kappa46 * ae * np.cos(np.deg2rad(alpha46))
v_x046 = - Ve(1, kappa46*ae) * np.sin(np.deg2rad(alpha46))
y046 = kappa46 * ae * np.sin(np.deg2rad(alpha46))
v_y046 = Ve(1, kappa46*ae) * np.cos(np.deg2rad(alpha46))

alpha47 = 116.94
kappa47 = 0.96
x047 = kappa47 * ae * np.cos(np.deg2rad(alpha47))
v_x047 = - Ve(1, kappa47*ae) * np.sin(np.deg2rad(alpha47))
y047 = kappa47 * ae * np.sin(np.deg2rad(alpha47))
v_y047 = Ve(1, kappa47*ae) * np.cos(np.deg2rad(alpha47))

alpha48 = 117.96
kappa48 = 0.97
x048 = kappa48 * ae * np.cos(np.deg2rad(alpha48))
v_x048 = - Ve(1, kappa48*ae) * np.sin(np.deg2rad(alpha48))
y048 = kappa48 * ae * np.sin(np.deg2rad(alpha48))
v_y048 = Ve(1, kappa48*ae) * np.cos(np.deg2rad(alpha48))

alpha49 = 118.98
kappa49 = 0.98
x049 = kappa49 * ae * np.cos(np.deg2rad(alpha49))
v_x049 = - Ve(1, kappa49*ae) * np.sin(np.deg2rad(alpha49))
y049 = kappa49 * ae * np.sin(np.deg2rad(alpha49))
v_y049 = Ve(1, kappa49*ae) * np.cos(np.deg2rad(alpha49))

alpha50 = 120.0
kappa50 = 1.0
x050 = kappa50 * ae * np.cos(np.deg2rad(alpha50))
v_x050 = - Ve(1, kappa50*ae) * np.sin(np.deg2rad(alpha50))
y050 = kappa50 * ae * np.sin(np.deg2rad(alpha50))
v_y050 = Ve(1, kappa50*ae) * np.cos(np.deg2rad(alpha50))


alpha51 = 250.0
kappa51 = 0.5
x051 = kappa51 * ae * np.cos(np.deg2rad(alpha51))
v_x051 = - Ve(1, kappa51*ae) * np.sin(np.deg2rad(alpha51))
y051 = kappa51 * ae * np.sin(np.deg2rad(alpha51))
v_y051 = Ve(1, kappa51*ae) * np.cos(np.deg2rad(alpha51))

alpha52 = 251.02
kappa52 = 0.51
x052 = kappa52 * ae * np.cos(np.deg2rad(alpha52))
v_x052 = - Ve(1, kappa52*ae) * np.sin(np.deg2rad(alpha52))
y052 = kappa52 * ae * np.sin(np.deg2rad(alpha52))
v_y052 = Ve(1, kappa52*ae) * np.cos(np.deg2rad(alpha52))

alpha53 = 252.04
kappa53 = 0.52
x053 = kappa53 * ae * np.cos(np.deg2rad(alpha53))
v_x053 = - Ve(1, kappa53*ae) * np.sin(np.deg2rad(alpha53))
y053 = kappa53 * ae * np.sin(np.deg2rad(alpha53))
v_y053 = Ve(1, kappa53*ae) * np.cos(np.deg2rad(alpha53))

alpha54 = 253.06
kappa54 = 0.53
x054 = kappa54 * ae * np.cos(np.deg2rad(alpha54))
v_x054 = - Ve(1, kappa54*ae) * np.sin(np.deg2rad(alpha54))
y054 = kappa54 * ae * np.sin(np.deg2rad(alpha54))
v_y054 = Ve(1, kappa54*ae) * np.cos(np.deg2rad(alpha54))

alpha55 = 254.08
kappa55 = 0.54
x055 = kappa55 * ae * np.cos(np.deg2rad(alpha55))
v_x055 = - Ve(1, kappa55*ae) * np.sin(np.deg2rad(alpha55))
y055 = kappa55 * ae * np.sin(np.deg2rad(alpha55))
v_y055 = Ve(1, kappa55*ae) * np.cos(np.deg2rad(alpha55))

alpha56 = 255.1
kappa56 = 0.55
x056 = kappa56 * ae * np.cos(np.deg2rad(alpha56))
v_x056 = - Ve(1, kappa56*ae) * np.sin(np.deg2rad(alpha56))
y056 = kappa56 * ae * np.sin(np.deg2rad(alpha56))
v_y056 = Ve(1, kappa56*ae) * np.cos(np.deg2rad(alpha56))

alpha57 = 256.12
kappa57 = 0.56
x057 = kappa57 * ae * np.cos(np.deg2rad(alpha57))
v_x057 = - Ve(1, kappa57*ae) * np.sin(np.deg2rad(alpha57))
y057 = kappa57 * ae * np.sin(np.deg2rad(alpha57))
v_y057 = Ve(1, kappa57*ae) * np.cos(np.deg2rad(alpha57))

alpha58 = 257.14
kappa58 = 0.57
x058 = kappa58 * ae * np.cos(np.deg2rad(alpha58))
v_x058 = - Ve(1, kappa58*ae) * np.sin(np.deg2rad(alpha58))
y058 = kappa58 * ae * np.sin(np.deg2rad(alpha58))
v_y058 = Ve(1, kappa58*ae) * np.cos(np.deg2rad(alpha58))

alpha59 = 258.16
kappa59 = 0.58
x059 = kappa59 * ae * np.cos(np.deg2rad(alpha59))
v_x059 = - Ve(1, kappa59*ae) * np.sin(np.deg2rad(alpha59))
y059 = kappa59 * ae * np.sin(np.deg2rad(alpha59))
v_y059 = Ve(1, kappa59*ae) * np.cos(np.deg2rad(alpha59))

alpha60 = 259.18
kappa60 = 0.59
x060 = kappa60 * ae * np.cos(np.deg2rad(alpha60))
v_x060 = - Ve(1, kappa60*ae) * np.sin(np.deg2rad(alpha60))
y060 = kappa60 * ae * np.sin(np.deg2rad(alpha60))
v_y060 = Ve(1, kappa60*ae) * np.cos(np.deg2rad(alpha60))

alpha61 = 260.2
kappa61 = 0.6
x061 = kappa61 * ae * np.cos(np.deg2rad(alpha61))
v_x061 = - Ve(1, kappa61*ae) * np.sin(np.deg2rad(alpha61))
y061 = kappa61 * ae * np.sin(np.deg2rad(alpha61))
v_y061 = Ve(1, kappa61*ae) * np.cos(np.deg2rad(alpha61))

alpha62 = 261.22
kappa62 = 0.61
x062 = kappa62 * ae * np.cos(np.deg2rad(alpha62))
v_x062 = - Ve(1, kappa62*ae) * np.sin(np.deg2rad(alpha62))
y062 = kappa62 * ae * np.sin(np.deg2rad(alpha62))
v_y062 = Ve(1, kappa62*ae) * np.cos(np.deg2rad(alpha62))

alpha63 = 262.24
kappa63 = 0.62
x063 = kappa63 * ae * np.cos(np.deg2rad(alpha63))
v_x063 = - Ve(1, kappa63*ae) * np.sin(np.deg2rad(alpha63))
y063 = kappa63 * ae * np.sin(np.deg2rad(alpha63))
v_y063 = Ve(1, kappa63*ae) * np.cos(np.deg2rad(alpha63))

alpha64 = 263.27
kappa64 = 0.63
x064 = kappa64 * ae * np.cos(np.deg2rad(alpha64))
v_x064 = - Ve(1, kappa64*ae) * np.sin(np.deg2rad(alpha64))
y064 = kappa64 * ae * np.sin(np.deg2rad(alpha64))
v_y064 = Ve(1, kappa64*ae) * np.cos(np.deg2rad(alpha64))

alpha65 = 264.29
kappa65 = 0.64
x065 = kappa65 * ae * np.cos(np.deg2rad(alpha65))
v_x065 = - Ve(1, kappa65*ae) * np.sin(np.deg2rad(alpha65))
y065 = kappa65 * ae * np.sin(np.deg2rad(alpha65))
v_y065 = Ve(1, kappa65*ae) * np.cos(np.deg2rad(alpha65))

alpha66 = 265.31
kappa66 = 0.65
x066 = kappa66 * ae * np.cos(np.deg2rad(alpha66))
v_x066 = - Ve(1, kappa66*ae) * np.sin(np.deg2rad(alpha66))
y066 = kappa66 * ae * np.sin(np.deg2rad(alpha66))
v_y066 = Ve(1, kappa66*ae) * np.cos(np.deg2rad(alpha66))

alpha67 = 266.33
kappa67 = 0.66
x067 = kappa67 * ae * np.cos(np.deg2rad(alpha67))
v_x067 = - Ve(1, kappa67*ae) * np.sin(np.deg2rad(alpha67))
y067 = kappa67 * ae * np.sin(np.deg2rad(alpha67))
v_y067 = Ve(1, kappa67*ae) * np.cos(np.deg2rad(alpha67))

alpha68 = 267.35
kappa68 = 0.67
x068 = kappa68 * ae * np.cos(np.deg2rad(alpha68))
v_x068 = - Ve(1, kappa68*ae) * np.sin(np.deg2rad(alpha68))
y068 = kappa68 * ae * np.sin(np.deg2rad(alpha68))
v_y068 = Ve(1, kappa68*ae) * np.cos(np.deg2rad(alpha68))

alpha69 = 268.37
kappa69 = 0.68
x069 = kappa69 * ae * np.cos(np.deg2rad(alpha69))
v_x069 = - Ve(1, kappa69*ae) * np.sin(np.deg2rad(alpha69))
y069 = kappa69 * ae * np.sin(np.deg2rad(alpha69))
v_y069 = Ve(1, kappa69*ae) * np.cos(np.deg2rad(alpha69))

alpha70 = 269.39
kappa70 = 0.69
x070 = kappa70 * ae * np.cos(np.deg2rad(alpha70))
v_x070 = - Ve(1, kappa70*ae) * np.sin(np.deg2rad(alpha70))
y070 = kappa70 * ae * np.sin(np.deg2rad(alpha70))
v_y070 = Ve(1, kappa70*ae) * np.cos(np.deg2rad(alpha70))

alpha71 = 270.41
kappa71 = 0.7
x071 = kappa71 * ae * np.cos(np.deg2rad(alpha71))
v_x071 = - Ve(1, kappa71*ae) * np.sin(np.deg2rad(alpha71))
y071 = kappa71 * ae * np.sin(np.deg2rad(alpha71))
v_y071 = Ve(1, kappa71*ae) * np.cos(np.deg2rad(alpha71))

alpha72 = 271.43
kappa72 = 0.71
x072 = kappa72 * ae * np.cos(np.deg2rad(alpha72))
v_x072 = - Ve(1, kappa72*ae) * np.sin(np.deg2rad(alpha72))
y072 = kappa72 * ae * np.sin(np.deg2rad(alpha72))
v_y072 = Ve(1, kappa72*ae) * np.cos(np.deg2rad(alpha72))

alpha73 = 272.45
kappa73 = 0.72
x073 = kappa73 * ae * np.cos(np.deg2rad(alpha73))
v_x073 = - Ve(1, kappa73*ae) * np.sin(np.deg2rad(alpha73))
y073 = kappa73 * ae * np.sin(np.deg2rad(alpha73))
v_y073 = Ve(1, kappa73*ae) * np.cos(np.deg2rad(alpha73))

alpha74 = 273.47
kappa74 = 0.73
x074 = kappa74 * ae * np.cos(np.deg2rad(alpha74))
v_x074 = - Ve(1, kappa74*ae) * np.sin(np.deg2rad(alpha74))
y074 = kappa74 * ae * np.sin(np.deg2rad(alpha74))
v_y074 = Ve(1, kappa74*ae) * np.cos(np.deg2rad(alpha74))

alpha75 = 274.49
kappa75 = 0.74
x075 = kappa75 * ae * np.cos(np.deg2rad(alpha75))
v_x075 = - Ve(1, kappa75*ae) * np.sin(np.deg2rad(alpha75))
y075 = kappa75 * ae * np.sin(np.deg2rad(alpha75))
v_y075 = Ve(1, kappa75*ae) * np.cos(np.deg2rad(alpha75))

alpha76 = 275.51
kappa76 = 0.75
x076 = kappa76 * ae * np.cos(np.deg2rad(alpha76))
v_x076 = - Ve(1, kappa76*ae) * np.sin(np.deg2rad(alpha76))
y076 = kappa76 * ae * np.sin(np.deg2rad(alpha76))
v_y076 = Ve(1, kappa76*ae) * np.cos(np.deg2rad(alpha76))

alpha77 = 276.53
kappa77 = 0.76
x077 = kappa77 * ae * np.cos(np.deg2rad(alpha77))
v_x077 = - Ve(1, kappa77*ae) * np.sin(np.deg2rad(alpha77))
y077 = kappa77 * ae * np.sin(np.deg2rad(alpha77))
v_y077 = Ve(1, kappa77*ae) * np.cos(np.deg2rad(alpha77))

alpha78 = 277.55
kappa78 = 0.77
x078 = kappa78 * ae * np.cos(np.deg2rad(alpha78))
v_x078 = - Ve(1, kappa78*ae) * np.sin(np.deg2rad(alpha78))
y078 = kappa78 * ae * np.sin(np.deg2rad(alpha78))
v_y078 = Ve(1, kappa78*ae) * np.cos(np.deg2rad(alpha78))

alpha79 = 278.57
kappa79 = 0.78
x079 = kappa79 * ae * np.cos(np.deg2rad(alpha79))
v_x079 = - Ve(1, kappa79*ae) * np.sin(np.deg2rad(alpha79))
y079 = kappa79 * ae * np.sin(np.deg2rad(alpha79))
v_y079 = Ve(1, kappa79*ae) * np.cos(np.deg2rad(alpha79))

alpha80 = 279.59
kappa80 = 0.79
x080 = kappa80 * ae * np.cos(np.deg2rad(alpha80))
v_x080 = - Ve(1, kappa80*ae) * np.sin(np.deg2rad(alpha80))
y080 = kappa80 * ae * np.sin(np.deg2rad(alpha80))
v_y080 = Ve(1, kappa80*ae) * np.cos(np.deg2rad(alpha80))

alpha81 = 280.61
kappa81 = 0.8
x081 = kappa81 * ae * np.cos(np.deg2rad(alpha81))
v_x081 = - Ve(1, kappa81*ae) * np.sin(np.deg2rad(alpha81))
y081 = kappa81 * ae * np.sin(np.deg2rad(alpha81))
v_y081 = Ve(1, kappa81*ae) * np.cos(np.deg2rad(alpha81))

alpha82 = 281.63
kappa82 = 0.81
x082 = kappa82 * ae * np.cos(np.deg2rad(alpha82))
v_x082 = - Ve(1, kappa82*ae) * np.sin(np.deg2rad(alpha82))
y082 = kappa82 * ae * np.sin(np.deg2rad(alpha82))
v_y082 = Ve(1, kappa82*ae) * np.cos(np.deg2rad(alpha82))

alpha83 = 282.65
kappa83 = 0.82
x083 = kappa83 * ae * np.cos(np.deg2rad(alpha83))
v_x083 = - Ve(1, kappa83*ae) * np.sin(np.deg2rad(alpha83))
y083 = kappa83 * ae * np.sin(np.deg2rad(alpha83))
v_y083 = Ve(1, kappa83*ae) * np.cos(np.deg2rad(alpha83))

alpha84 = 283.67
kappa84 = 0.83
x084 = kappa84 * ae * np.cos(np.deg2rad(alpha84))
v_x084 = - Ve(1, kappa84*ae) * np.sin(np.deg2rad(alpha84))
y084 = kappa84 * ae * np.sin(np.deg2rad(alpha84))
v_y084 = Ve(1, kappa84*ae) * np.cos(np.deg2rad(alpha84))

alpha85 = 284.69
kappa85 = 0.84
x085 = kappa85 * ae * np.cos(np.deg2rad(alpha85))
v_x085 = - Ve(1, kappa85*ae) * np.sin(np.deg2rad(alpha85))
y085 = kappa85 * ae * np.sin(np.deg2rad(alpha85))
v_y085 = Ve(1, kappa85*ae) * np.cos(np.deg2rad(alpha85))

alpha86 = 285.71
kappa86 = 0.85
x086 = kappa86 * ae * np.cos(np.deg2rad(alpha86))
v_x086 = - Ve(1, kappa86*ae) * np.sin(np.deg2rad(alpha86))
y086 = kappa86 * ae * np.sin(np.deg2rad(alpha86))
v_y086 = Ve(1, kappa86*ae) * np.cos(np.deg2rad(alpha86))

alpha87 = 286.73
kappa87 = 0.86
x087 = kappa87 * ae * np.cos(np.deg2rad(alpha87))
v_x087 = - Ve(1, kappa87*ae) * np.sin(np.deg2rad(alpha87))
y087 = kappa87 * ae * np.sin(np.deg2rad(alpha87))
v_y087 = Ve(1, kappa87*ae) * np.cos(np.deg2rad(alpha87))

alpha88 = 287.76
kappa88 = 0.87
x088 = kappa88 * ae * np.cos(np.deg2rad(alpha88))
v_x088 = - Ve(1, kappa88*ae) * np.sin(np.deg2rad(alpha88))
y088 = kappa88 * ae * np.sin(np.deg2rad(alpha88))
v_y088 = Ve(1, kappa88*ae) * np.cos(np.deg2rad(alpha88))

alpha89 = 288.78
kappa89 = 0.88
x089 = kappa89 * ae * np.cos(np.deg2rad(alpha89))
v_x089 = - Ve(1, kappa89*ae) * np.sin(np.deg2rad(alpha89))
y089 = kappa89 * ae * np.sin(np.deg2rad(alpha89))
v_y089 = Ve(1, kappa89*ae) * np.cos(np.deg2rad(alpha89))

alpha90 = 289.8
kappa90 = 0.89
x090 = kappa90 * ae * np.cos(np.deg2rad(alpha90))
v_x090 = - Ve(1, kappa90*ae) * np.sin(np.deg2rad(alpha90))
y090 = kappa90 * ae * np.sin(np.deg2rad(alpha90))
v_y090 = Ve(1, kappa90*ae) * np.cos(np.deg2rad(alpha90))

alpha91 = 290.82
kappa91 = 0.9
x091 = kappa91 * ae * np.cos(np.deg2rad(alpha91))
v_x091 = - Ve(1, kappa91*ae) * np.sin(np.deg2rad(alpha91))
y091 = kappa91 * ae * np.sin(np.deg2rad(alpha91))
v_y091 = Ve(1, kappa91*ae) * np.cos(np.deg2rad(alpha91))

alpha92 = 291.84
kappa92 = 0.91
x092 = kappa92 * ae * np.cos(np.deg2rad(alpha92))
v_x092 = - Ve(1, kappa92*ae) * np.sin(np.deg2rad(alpha92))
y092 = kappa92 * ae * np.sin(np.deg2rad(alpha92))
v_y092 = Ve(1, kappa92*ae) * np.cos(np.deg2rad(alpha92))

alpha93 = 292.86
kappa93 = 0.92
x093 = kappa93 * ae * np.cos(np.deg2rad(alpha93))
v_x093 = - Ve(1, kappa93*ae) * np.sin(np.deg2rad(alpha93))
y093 = kappa93 * ae * np.sin(np.deg2rad(alpha93))
v_y093 = Ve(1, kappa93*ae) * np.cos(np.deg2rad(alpha93))

alpha94 = 293.88
kappa94 = 0.93
x094 = kappa94 * ae * np.cos(np.deg2rad(alpha94))
v_x094 = - Ve(1, kappa94*ae) * np.sin(np.deg2rad(alpha94))
y094 = kappa94 * ae * np.sin(np.deg2rad(alpha94))
v_y094 = Ve(1, kappa94*ae) * np.cos(np.deg2rad(alpha94))

alpha95 = 294.9
kappa95 = 0.94
x095 = kappa95 * ae * np.cos(np.deg2rad(alpha95))
v_x095 = - Ve(1, kappa95*ae) * np.sin(np.deg2rad(alpha95))
y095 = kappa95 * ae * np.sin(np.deg2rad(alpha95))
v_y095 = Ve(1, kappa95*ae) * np.cos(np.deg2rad(alpha95))

alpha96 = 295.92
kappa96 = 0.95
x096 = kappa96 * ae * np.cos(np.deg2rad(alpha96))
v_x096 = - Ve(1, kappa96*ae) * np.sin(np.deg2rad(alpha96))
y096 = kappa96 * ae * np.sin(np.deg2rad(alpha96))
v_y096 = Ve(1, kappa96*ae) * np.cos(np.deg2rad(alpha96))

alpha97 = 296.94
kappa97 = 0.96
x097 = kappa97 * ae * np.cos(np.deg2rad(alpha97))
v_x097 = - Ve(1, kappa97*ae) * np.sin(np.deg2rad(alpha97))
y097 = kappa97 * ae * np.sin(np.deg2rad(alpha97))
v_y097 = Ve(1, kappa97*ae) * np.cos(np.deg2rad(alpha97))

alpha98 = 297.96
kappa98 = 0.97
x098 = kappa98 * ae * np.cos(np.deg2rad(alpha98))
v_x098 = - Ve(1, kappa98*ae) * np.sin(np.deg2rad(alpha98))
y098 = kappa98 * ae * np.sin(np.deg2rad(alpha98))
v_y098 = Ve(1, kappa98*ae) * np.cos(np.deg2rad(alpha98))

alpha99 = 298.98
kappa99 = 0.98
x099 = kappa99 * ae * np.cos(np.deg2rad(alpha99))
v_x099 = - Ve(1, kappa99*ae) * np.sin(np.deg2rad(alpha99))
y099 = kappa99 * ae * np.sin(np.deg2rad(alpha99))
v_y099 = Ve(1, kappa99*ae) * np.cos(np.deg2rad(alpha99))

alpha100 = 300.0
kappa100 = 1.0
x0100 = kappa100 * ae * np.cos(np.deg2rad(alpha100))
v_x0100 = - Ve(1, kappa100*ae) * np.sin(np.deg2rad(alpha100))
y0100 = kappa100 * ae * np.sin(np.deg2rad(alpha100))
v_y0100 = Ve(1, kappa100*ae) * np.cos(np.deg2rad(alpha100))

center_radius1 = 0.5 * ae
angular_speed = 2 * np.pi / (0.5 * seconds_in_year)

angle1 = 0
xc1 = center_radius1 * np.cos(np.deg2rad(angle1))
yc1 = center_radius1 * np.sin(np.deg2rad(angle1))
v_xc1 = -angular_speed * yc1
v_yc1 = angular_speed * xc1

angle2 = 1
xc2 = center_radius1 * np.cos(np.deg2rad(angle2))
yc2 = center_radius1 * np.sin(np.deg2rad(angle2))
v_xc2 = -angular_speed * yc2
v_yc2 = angular_speed * xc2

angle3 = 2
xc3 = center_radius1 * np.cos(np.deg2rad(angle3))
yc3 = center_radius1 * np.sin(np.deg2rad(angle3))
v_xc3 = -angular_speed * yc3
v_yc3 = angular_speed * xc3

angle4 = 3
xc4 = center_radius1 * np.cos(np.deg2rad(angle4))
yc4 = center_radius1 * np.sin(np.deg2rad(angle4))
v_xc4 = -angular_speed * yc4
v_yc4 = angular_speed * xc4

angle5 = 4
xc5 = center_radius1 * np.cos(np.deg2rad(angle5))
yc5 = center_radius1 * np.sin(np.deg2rad(angle5))
v_xc5 = -angular_speed * yc5
v_yc5 = angular_speed * xc5

angle6 = 5
xc6 = center_radius1 * np.cos(np.deg2rad(angle6))
yc6 = center_radius1 * np.sin(np.deg2rad(angle6))
v_xc6 = -angular_speed * yc6
v_yc6 = angular_speed * xc6

angle7 = 6
xc7 = center_radius1 * np.cos(np.deg2rad(angle7))
yc7 = center_radius1 * np.sin(np.deg2rad(angle7))
v_xc7 = -angular_speed * yc7
v_yc7 = angular_speed * xc7

angle8 = 7
xc8 = center_radius1 * np.cos(np.deg2rad(angle8))
yc8 = center_radius1 * np.sin(np.deg2rad(angle8))
v_xc8 = -angular_speed * yc8
v_yc8 = angular_speed * xc8

angle9 = 8
xc9 = center_radius1 * np.cos(np.deg2rad(angle9))
yc9 = center_radius1 * np.sin(np.deg2rad(angle9))
v_xc9 = -angular_speed * yc9
v_yc9 = angular_speed * xc9

angle10 = 9
xc10 = center_radius1 * np.cos(np.deg2rad(angle10))
yc10 = center_radius1 * np.sin(np.deg2rad(angle10))
v_xc10 = -angular_speed * yc10
v_yc10 = angular_speed * xc10

angle11 = 10
xc11 = center_radius1 * np.cos(np.deg2rad(angle11))
yc11 = center_radius1 * np.sin(np.deg2rad(angle11))
v_xc11 = -angular_speed * yc11
v_yc11 = angular_speed * xc11

angle12 = 11
xc12 = center_radius1 * np.cos(np.deg2rad(angle12))
yc12 = center_radius1 * np.sin(np.deg2rad(angle12))
v_xc12 = -angular_speed * yc12
v_yc12 = angular_speed * xc12

angle13 = 12
xc13 = center_radius1 * np.cos(np.deg2rad(angle13))
yc13 = center_radius1 * np.sin(np.deg2rad(angle13))
v_xc13 = -angular_speed * yc13
v_yc13 = angular_speed * xc13

angle14 = 13
xc14 = center_radius1 * np.cos(np.deg2rad(angle14))
yc14 = center_radius1 * np.sin(np.deg2rad(angle14))
v_xc14 = -angular_speed * yc14
v_yc14 = angular_speed * xc14

angle15 = 14
xc15 = center_radius1 * np.cos(np.deg2rad(angle15))
yc15 = center_radius1 * np.sin(np.deg2rad(angle15))
v_xc15 = -angular_speed * yc15
v_yc15 = angular_speed * xc15

angle16 = 15
xc16 = center_radius1 * np.cos(np.deg2rad(angle16))
yc16 = center_radius1 * np.sin(np.deg2rad(angle16))
v_xc16 = -angular_speed * yc16
v_yc16 = angular_speed * xc16

angle17 = 16
xc17 = center_radius1 * np.cos(np.deg2rad(angle17))
yc17 = center_radius1 * np.sin(np.deg2rad(angle17))
v_xc17 = -angular_speed * yc17
v_yc17 = angular_speed * xc17

angle18 = 17
xc18 = center_radius1 * np.cos(np.deg2rad(angle18))
yc18 = center_radius1 * np.sin(np.deg2rad(angle18))
v_xc18 = -angular_speed * yc18
v_yc18 = angular_speed * xc18

center_radius2 = 0.5 * ae

angle19 = 1
xc19 = center_radius2 * np.cos(np.deg2rad(angle19))
yc19 = center_radius2 * np.sin(np.deg2rad(angle19))
v_xc19 = -angular_speed * yc19
v_yc19 = angular_speed * xc19

angle20 = 2
xc20 = center_radius2 * np.cos(np.deg2rad(angle20))
yc20 = center_radius2 * np.sin(np.deg2rad(angle20))
v_xc20 = -angular_speed * yc20
v_yc20 = angular_speed * xc20

angle21 = 3
xc21 = center_radius2 * np.cos(np.deg2rad(angle21))
yc21 = center_radius2 * np.sin(np.deg2rad(angle21))
v_xc21 = -angular_speed * yc21
v_yc21 = angular_speed * xc21

angle22 = 4
xc22 = center_radius2 * np.cos(np.deg2rad(angle22))
yc22 = center_radius2 * np.sin(np.deg2rad(angle22))
v_xc22 = -angular_speed * yc22
v_yc22 = angular_speed * xc22

angle23 = 5
xc23 = center_radius2 * np.cos(np.deg2rad(angle23))
yc23 = center_radius2 * np.sin(np.deg2rad(angle23))
v_xc23 = -angular_speed * yc23
v_yc23 = angular_speed * xc23

angle24 = 6
xc24 = center_radius2 * np.cos(np.deg2rad(angle24))
yc24 = center_radius2 * np.sin(np.deg2rad(angle24))
v_xc24 = -angular_speed * yc24
v_yc24 = angular_speed * xc24

angle25 = 7
xc25 = center_radius2 * np.cos(np.deg2rad(angle25))
yc25 = center_radius2 * np.sin(np.deg2rad(angle25))
v_xc25 = -angular_speed * yc25
v_yc25 = angular_speed * xc25

angle26 = 8
xc26 = center_radius2 * np.cos(np.deg2rad(angle26))
yc26 = center_radius2 * np.sin(np.deg2rad(angle26))
v_xc26 = -angular_speed * yc26
v_yc26 = angular_speed * xc26

angle27 = 9
xc27 = center_radius2 * np.cos(np.deg2rad(angle27))
yc27 = center_radius2 * np.sin(np.deg2rad(angle27))
v_xc27 = -angular_speed * yc27
v_yc27 = angular_speed * xc27

angle28 = 10
xc28 = center_radius2 * np.cos(np.deg2rad(angle28))
yc28 = center_radius2 * np.sin(np.deg2rad(angle28))
v_xc28 = -angular_speed * yc28
v_yc28 = angular_speed * xc28

angle29 = 11
xc29 = center_radius2 * np.cos(np.deg2rad(angle29))
yc29 = center_radius2 * np.sin(np.deg2rad(angle29))
v_xc29 = -angular_speed * yc29
v_yc29 = angular_speed * xc29

angle30 = 12
xc30 = center_radius2 * np.cos(np.deg2rad(angle30))
yc30 = center_radius2 * np.sin(np.deg2rad(angle30))
v_xc30 = -angular_speed * yc30
v_yc30 = angular_speed * xc30

angle31 = 13
xc31 = center_radius2 * np.cos(np.deg2rad(angle31))
yc31 = center_radius2 * np.sin(np.deg2rad(angle31))
v_xc31 = -angular_speed * yc31
v_yc31 = angular_speed * xc31

angle32 = 14
xc32 = center_radius2 * np.cos(np.deg2rad(angle32))
yc32 = center_radius2 * np.sin(np.deg2rad(angle32))
v_xc32 = -angular_speed * yc32
v_yc32 = angular_speed * xc32

angle33 = 15
xc33 = center_radius2 * np.cos(np.deg2rad(angle33))
yc33 = center_radius2 * np.sin(np.deg2rad(angle33))
v_xc33 = -angular_speed * yc33
v_yc33 = angular_speed * xc33

angle34 = 16
xc34 = center_radius2 * np.cos(np.deg2rad(angle34))
yc34 = center_radius2 * np.sin(np.deg2rad(angle34))
v_xc34 = -angular_speed * yc34
v_yc34 = angular_speed * xc34

angle35 = 17
xc35 = center_radius2 * np.cos(np.deg2rad(angle35))
yc35 = center_radius2 * np.sin(np.deg2rad(angle35))
v_xc35 = -angular_speed * yc35
v_yc35 = angular_speed * xc35


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
      x023, v_x023, y023, v_y023,
      x024, v_x024, y024, v_y024,
      x025, v_x025, y025, v_y025,
      x026, v_x026, y026, v_y026,
      x027, v_x027, y027, v_y027,
      x028, v_x028, y028, v_y028,
      x029, v_x029, y029, v_y029,
      x030, v_x030, y030, v_y030,
      x031, v_x031, y031, v_y031,
      x032, v_x032, y032, v_y032,
      x033, v_x033, y033, v_y033,
      x034, v_x034, y034, v_y034,
      x035, v_x035, y035, v_y035,
      x036, v_x036, y036, v_y036,
      x037, v_x037, y037, v_y037,
      x038, v_x038, y038, v_y038,
      x039, v_x039, y039, v_y039,
      x040, v_x040, y040, v_y040,
      x041, v_x041, y041, v_y041,
      x042, v_x042, y042, v_y042,
      x043, v_x043, y043, v_y043,
      x044, v_x044, y044, v_y044,
      x045, v_x045, y045, v_y045,
      x046, v_x046, y046, v_y046,
      x047, v_x047, y047, v_y047,
      x048, v_x048, y048, v_y048,
      x049, v_x049, y049, v_y049,
      x050, v_x050, y050, v_y050,
      x051, v_x051, y051, v_y051,
      x052, v_x052, y052, v_y052,
      x053, v_x053, y053, v_y053,
      x054, v_x054, y054, v_y054,
      x055, v_x055, y055, v_y055,
      x056, v_x056, y056, v_y056,
      x057, v_x057, y057, v_y057,
      x058, v_x058, y058, v_y058,
      x059, v_x059, y059, v_y059,
      x060, v_x060, y060, v_y060,
      x061, v_x061, y061, v_y061,
      x062, v_x062, y062, v_y062,
      x063, v_x063, y063, v_y063,
      x064, v_x064, y064, v_y064,
      x065, v_x065, y065, v_y065,
      x066, v_x066, y066, v_y066,
      x067, v_x067, y067, v_y067,
      x068, v_x068, y068, v_y068,
      x069, v_x069, y069, v_y069,
      x070, v_x070, y070, v_y070,
      x071, v_x071, y071, v_y071,
      x072, v_x072, y072, v_y072,
      x073, v_x073, y073, v_y073,
      x074, v_x074, y074, v_y074,
      x075, v_x075, y075, v_y075,
      x076, v_x076, y076, v_y076,
      x077, v_x077, y077, v_y077,
      x078, v_x078, y078, v_y078,
      x079, v_x079, y079, v_y079,
      x080, v_x080, y080, v_y080,
      x081, v_x081, y081, v_y081,
      x082, v_x082, y082, v_y082,
      x083, v_x083, y083, v_y083,
      x084, v_x084, y084, v_y084,
      x085, v_x085, y085, v_y085,
      x086, v_x086, y086, v_y086,
      x087, v_x087, y087, v_y087,
      x088, v_x088, y088, v_y088,
      x089, v_x089, y089, v_y089,
      x090, v_x090, y090, v_y090,
      x091, v_x091, y091, v_y091,
      x092, v_x092, y092, v_y092,
      x093, v_x093, y093, v_y093,
      x094, v_x094, y094, v_y094,
      x095, v_x095, y095, v_y095,
      x096, v_x096, y096, v_y096,
      x097, v_x097, y097, v_y097,
      x098, v_x098, y098, v_y098,
      x099, v_x099, y099, v_y099,
      x0100, v_x0100, y0100, v_y0100,
      xc1, v_xc1, yc1, v_yc1,
      xc2, v_xc2, yc2, v_yc2,
      xc3, v_xc3, yc3, v_yc3,
      xc4, v_xc4, yc4, v_yc4,
      xc5, v_xc5, yc5, v_yc5,
      xc6, v_xc6, yc6, v_yc6,
      xc7, v_xc7, yc7, v_yc7,
      xc8, v_xc8, yc8, v_yc8,
      xc9, v_xc9, yc9, v_yc9,
      xc10, v_xc10, yc10, v_yc10,
      xc11, v_xc11, yc11, v_yc11,
      xc12, v_xc12, yc12, v_yc12,
      xc13, v_xc13, yc13, v_yc13,
      xc14, v_xc14, yc14, v_yc14,
      xc15, v_xc15, yc15, v_yc15,
      xc16, v_xc16, yc16, v_yc16,
      xc17, v_xc17, yc17, v_yc17,
      xc18, v_xc18, yc18, v_yc18,
      xc19, v_xc19, yc19, v_yc19,
      xc20, v_xc20, yc20, v_yc20,
      xc21, v_xc21, yc21, v_yc21,
      xc22, v_xc22, yc22, v_yc22,
      xc23, v_xc23, yc23, v_yc23,
      xc24, v_xc24, yc24, v_yc24,
      xc25, v_xc25, yc25, v_yc25,
      xc26, v_xc26, yc26, v_yc26,
      xc27, v_xc27, yc27, v_yc27,
      xc28, v_xc28, yc28, v_yc28,
      xc29, v_xc29, yc29, v_yc29,
      xc30, v_xc30, yc30, v_yc30,
      xc31, v_xc31, yc31, v_yc31,
      xc32, v_xc32, yc32, v_yc32,
      xc33, v_xc33, yc33, v_yc33,
      xc34, v_xc34, yc34, v_yc34,
      xc35, v_xc35, yc35, v_yc35)

sol = odeint(move_func, s0, t)

# Строим решение в виде графика и анимируем
fig, ax = plt.subplots()
ax.set_facecolor('k')

number_points = 135
points = []
points_lines = []

for i in range(number_points):
    # points.append(plt.plot([], [], ',', color='r'))
    points.append(plt.plot([], [], 'o', color='w', ms='0.5'))
    # points_lines.append(plt.plot([], [], '-', color='r'))

def animate(i):
    for j in range(number_points):
        # Исправление: передаем точки как списки с одним элементом
        points[j][0].set_data([sol[i, 4 * j]], [sol[i, 4 * j + 2]])
        # points_lines[j][0].set_data(sol[:i, 4 * j], sol[:i, 4 * j + 2])

ani = FuncAnimation(fig, animate, frames=frames, interval=30)

plt.axis('equal')
ax.set_xlim(-edge, edge)
ax.set_ylim(-edge, edge)

# plt.plot([0], [0], 'o', color='w', ms=20)

ani.save('galaxy.gif', writer='pillow')