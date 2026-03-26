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
     x100, v_x100, y100, v_y100) = s
    
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

    dxdt23 = v_x23
    dv_xdt23 = - G * M * x23 / (x23**2 + y23**2)**1.5
    dydt23 = v_y23
    dv_ydt23 = - G * M * y23 / (x23**2 + y23**2)**1.5

    dxdt24 = v_x24
    dv_xdt24 = - G * M * x24 / (x24**2 + y24**2)**1.5
    dydt24 = v_y24
    dv_ydt24 = - G * M * y24 / (x24**2 + y24**2)**1.5

    dxdt25 = v_x25
    dv_xdt25 = - G * M * x25 / (x25**2 + y25**2)**1.5
    dydt25 = v_y25
    dv_ydt25 = - G * M * y25 / (x25**2 + y25**2)**1.5

    dxdt26 = v_x26
    dv_xdt26 = - G * M * x26 / (x26**2 + y26**2)**1.5
    dydt26 = v_y26
    dv_ydt26 = - G * M * y26 / (x26**2 + y26**2)**1.5

    dxdt27 = v_x27
    dv_xdt27 = - G * M * x27 / (x27**2 + y27**2)**1.5
    dydt27 = v_y27
    dv_ydt27 = - G * M * y27 / (x27**2 + y27**2)**1.5

    dxdt28 = v_x28
    dv_xdt28 = - G * M * x28 / (x28**2 + y28**2)**1.5
    dydt28 = v_y28
    dv_ydt28 = - G * M * y28 / (x28**2 + y28**2)**1.5

    dxdt29 = v_x29
    dv_xdt29 = - G * M * x29 / (x29**2 + y29**2)**1.5
    dydt29 = v_y29
    dv_ydt29 = - G * M * y29 / (x29**2 + y29**2)**1.5

    dxdt30 = v_x30
    dv_xdt30 = - G * M * x30 / (x30**2 + y30**2)**1.5
    dydt30 = v_y30
    dv_ydt30 = - G * M * y30 / (x30**2 + y30**2)**1.5

    dxdt31 = v_x31
    dv_xdt31 = - G * M * x31 / (x31**2 + y31**2)**1.5
    dydt31 = v_y31
    dv_ydt31 = - G * M * y31 / (x31**2 + y31**2)**1.5

    dxdt32 = v_x32
    dv_xdt32 = - G * M * x32 / (x32**2 + y32**2)**1.5
    dydt32 = v_y32
    dv_ydt32 = - G * M * y32 / (x32**2 + y32**2)**1.5

    dxdt33 = v_x33
    dv_xdt33 = - G * M * x33 / (x33**2 + y33**2)**1.5
    dydt33 = v_y33
    dv_ydt33 = - G * M * y33 / (x33**2 + y33**2)**1.5

    dxdt34 = v_x34
    dv_xdt34 = - G * M * x34 / (x34**2 + y34**2)**1.5
    dydt34 = v_y34
    dv_ydt34 = - G * M * y34 / (x34**2 + y34**2)**1.5

    dxdt35 = v_x35
    dv_xdt35 = - G * M * x35 / (x35**2 + y35**2)**1.5
    dydt35 = v_y35
    dv_ydt35 = - G * M * y35 / (x35**2 + y35**2)**1.5

    dxdt36 = v_x36
    dv_xdt36 = - G * M * x36 / (x36**2 + y36**2)**1.5
    dydt36 = v_y36
    dv_ydt36 = - G * M * y36 / (x36**2 + y36**2)**1.5

    dxdt37 = v_x37
    dv_xdt37 = - G * M * x37 / (x37**2 + y37**2)**1.5
    dydt37 = v_y37
    dv_ydt37 = - G * M * y37 / (x37**2 + y37**2)**1.5

    dxdt38 = v_x38
    dv_xdt38 = - G * M * x38 / (x38**2 + y38**2)**1.5
    dydt38 = v_y38
    dv_ydt38 = - G * M * y38 / (x38**2 + y38**2)**1.5

    dxdt39 = v_x39
    dv_xdt39 = - G * M * x39 / (x39**2 + y39**2)**1.5
    dydt39 = v_y39
    dv_ydt39 = - G * M * y39 / (x39**2 + y39**2)**1.5

    dxdt40 = v_x40
    dv_xdt40 = - G * M * x40 / (x40**2 + y40**2)**1.5
    dydt40 = v_y40
    dv_ydt40 = - G * M * y40 / (x40**2 + y40**2)**1.5

    dxdt41 = v_x41
    dv_xdt41 = - G * M * x41 / (x41**2 + y41**2)**1.5
    dydt41 = v_y41
    dv_ydt41 = - G * M * y41 / (x41**2 + y41**2)**1.5

    dxdt42 = v_x42
    dv_xdt42 = - G * M * x42 / (x42**2 + y42**2)**1.5
    dydt42 = v_y42
    dv_ydt42 = - G * M * y42 / (x42**2 + y42**2)**1.5

    dxdt43 = v_x43
    dv_xdt43 = - G * M * x43 / (x43**2 + y43**2)**1.5
    dydt43 = v_y43
    dv_ydt43 = - G * M * y43 / (x43**2 + y43**2)**1.5

    dxdt44 = v_x44
    dv_xdt44 = - G * M * x44 / (x44**2 + y44**2)**1.5
    dydt44 = v_y44
    dv_ydt44 = - G * M * y44 / (x44**2 + y44**2)**1.5

    dxdt45 = v_x45
    dv_xdt45 = - G * M * x45 / (x45**2 + y45**2)**1.5
    dydt45 = v_y45
    dv_ydt45 = - G * M * y45 / (x45**2 + y45**2)**1.5

    dxdt46 = v_x46
    dv_xdt46 = - G * M * x46 / (x46**2 + y46**2)**1.5
    dydt46 = v_y46
    dv_ydt46 = - G * M * y46 / (x46**2 + y46**2)**1.5

    dxdt47 = v_x47
    dv_xdt47 = - G * M * x47 / (x47**2 + y47**2)**1.5
    dydt47 = v_y47
    dv_ydt47 = - G * M * y47 / (x47**2 + y47**2)**1.5

    dxdt48 = v_x48
    dv_xdt48 = - G * M * x48 / (x48**2 + y48**2)**1.5
    dydt48 = v_y48
    dv_ydt48 = - G * M * y48 / (x48**2 + y48**2)**1.5

    dxdt49 = v_x49
    dv_xdt49 = - G * M * x49 / (x49**2 + y49**2)**1.5
    dydt49 = v_y49
    dv_ydt49 = - G * M * y49 / (x49**2 + y49**2)**1.5

    dxdt50 = v_x50
    dv_xdt50 = - G * M * x50 / (x50**2 + y50**2)**1.5
    dydt50 = v_y50
    dv_ydt50 = - G * M * y50 / (x50**2 + y50**2)**1.5

    dxdt51 = v_x51
    dv_xdt51 = - G * M * x51 / (x51**2 + y51**2)**1.5
    dydt51 = v_y51
    dv_ydt51 = - G * M * y51 / (x51**2 + y51**2)**1.5

    dxdt52 = v_x52
    dv_xdt52 = - G * M * x52 / (x52**2 + y52**2)**1.5
    dydt52 = v_y52
    dv_ydt52 = - G * M * y52 / (x52**2 + y52**2)**1.5

    dxdt53 = v_x53
    dv_xdt53 = - G * M * x53 / (x53**2 + y53**2)**1.5
    dydt53 = v_y53
    dv_ydt53 = - G * M * y53 / (x53**2 + y53**2)**1.5

    dxdt54 = v_x54
    dv_xdt54 = - G * M * x54 / (x54**2 + y54**2)**1.5
    dydt54 = v_y54
    dv_ydt54 = - G * M * y54 / (x54**2 + y54**2)**1.5

    dxdt55 = v_x55
    dv_xdt55 = - G * M * x55 / (x55**2 + y55**2)**1.5
    dydt55 = v_y55
    dv_ydt55 = - G * M * y55 / (x55**2 + y55**2)**1.5

    dxdt56 = v_x56
    dv_xdt56 = - G * M * x56 / (x56**2 + y56**2)**1.5
    dydt56 = v_y56
    dv_ydt56 = - G * M * y56 / (x56**2 + y56**2)**1.5

    dxdt57 = v_x57
    dv_xdt57 = - G * M * x57 / (x57**2 + y57**2)**1.5
    dydt57 = v_y57
    dv_ydt57 = - G * M * y57 / (x57**2 + y57**2)**1.5

    dxdt58 = v_x58
    dv_xdt58 = - G * M * x58 / (x58**2 + y58**2)**1.5
    dydt58 = v_y58
    dv_ydt58 = - G * M * y58 / (x58**2 + y58**2)**1.5

    dxdt59 = v_x59
    dv_xdt59 = - G * M * x59 / (x59**2 + y59**2)**1.5
    dydt59 = v_y59
    dv_ydt59 = - G * M * y59 / (x59**2 + y59**2)**1.5

    dxdt60 = v_x60
    dv_xdt60 = - G * M * x60 / (x60**2 + y60**2)**1.5
    dydt60 = v_y60
    dv_ydt60 = - G * M * y60 / (x60**2 + y60**2)**1.5

    dxdt61 = v_x61
    dv_xdt61 = - G * M * x61 / (x61**2 + y61**2)**1.5
    dydt61 = v_y61
    dv_ydt61 = - G * M * y61 / (x61**2 + y61**2)**1.5

    dxdt62 = v_x62
    dv_xdt62 = - G * M * x62 / (x62**2 + y62**2)**1.5
    dydt62 = v_y62
    dv_ydt62 = - G * M * y62 / (x62**2 + y62**2)**1.5

    dxdt63 = v_x63
    dv_xdt63 = - G * M * x63 / (x63**2 + y63**2)**1.5
    dydt63 = v_y63
    dv_ydt63 = - G * M * y63 / (x63**2 + y63**2)**1.5

    dxdt64 = v_x64
    dv_xdt64 = - G * M * x64 / (x64**2 + y64**2)**1.5
    dydt64 = v_y64
    dv_ydt64 = - G * M * y64 / (x64**2 + y64**2)**1.5

    dxdt65 = v_x65
    dv_xdt65 = - G * M * x65 / (x65**2 + y65**2)**1.5
    dydt65 = v_y65
    dv_ydt65 = - G * M * y65 / (x65**2 + y65**2)**1.5

    dxdt66 = v_x66
    dv_xdt66 = - G * M * x66 / (x66**2 + y66**2)**1.5
    dydt66 = v_y66
    dv_ydt66 = - G * M * y66 / (x66**2 + y66**2)**1.5

    dxdt67 = v_x67
    dv_xdt67 = - G * M * x67 / (x67**2 + y67**2)**1.5
    dydt67 = v_y67
    dv_ydt67 = - G * M * y67 / (x67**2 + y67**2)**1.5

    dxdt68 = v_x68
    dv_xdt68 = - G * M * x68 / (x68**2 + y68**2)**1.5
    dydt68 = v_y68
    dv_ydt68 = - G * M * y68 / (x68**2 + y68**2)**1.5

    dxdt69 = v_x69
    dv_xdt69 = - G * M * x69 / (x69**2 + y69**2)**1.5
    dydt69 = v_y69
    dv_ydt69 = - G * M * y69 / (x69**2 + y69**2)**1.5

    dxdt70 = v_x70
    dv_xdt70 = - G * M * x70 / (x70**2 + y70**2)**1.5
    dydt70 = v_y70
    dv_ydt70 = - G * M * y70 / (x70**2 + y70**2)**1.5

    dxdt71 = v_x71
    dv_xdt71 = - G * M * x71 / (x71**2 + y71**2)**1.5
    dydt71 = v_y71
    dv_ydt71 = - G * M * y71 / (x71**2 + y71**2)**1.5

    dxdt72 = v_x72
    dv_xdt72 = - G * M * x72 / (x72**2 + y72**2)**1.5
    dydt72 = v_y72
    dv_ydt72 = - G * M * y72 / (x72**2 + y72**2)**1.5

    dxdt73 = v_x73
    dv_xdt73 = - G * M * x73 / (x73**2 + y73**2)**1.5
    dydt73 = v_y73
    dv_ydt73 = - G * M * y73 / (x73**2 + y73**2)**1.5

    dxdt74 = v_x74
    dv_xdt74 = - G * M * x74 / (x74**2 + y74**2)**1.5
    dydt74 = v_y74
    dv_ydt74 = - G * M * y74 / (x74**2 + y74**2)**1.5

    dxdt75 = v_x75
    dv_xdt75 = - G * M * x75 / (x75**2 + y75**2)**1.5
    dydt75 = v_y75
    dv_ydt75 = - G * M * y75 / (x75**2 + y75**2)**1.5

    dxdt76 = v_x76
    dv_xdt76 = - G * M * x76 / (x76**2 + y76**2)**1.5
    dydt76 = v_y76
    dv_ydt76 = - G * M * y76 / (x76**2 + y76**2)**1.5

    dxdt77 = v_x77
    dv_xdt77 = - G * M * x77 / (x77**2 + y77**2)**1.5
    dydt77 = v_y77
    dv_ydt77 = - G * M * y77 / (x77**2 + y77**2)**1.5

    dxdt78 = v_x78
    dv_xdt78 = - G * M * x78 / (x78**2 + y78**2)**1.5
    dydt78 = v_y78
    dv_ydt78 = - G * M * y78 / (x78**2 + y78**2)**1.5

    dxdt79 = v_x79
    dv_xdt79 = - G * M * x79 / (x79**2 + y79**2)**1.5
    dydt79 = v_y79
    dv_ydt79 = - G * M * y79 / (x79**2 + y79**2)**1.5

    dxdt80 = v_x80
    dv_xdt80 = - G * M * x80 / (x80**2 + y80**2)**1.5
    dydt80 = v_y80
    dv_ydt80 = - G * M * y80 / (x80**2 + y80**2)**1.5

    dxdt81 = v_x81
    dv_xdt81 = - G * M * x81 / (x81**2 + y81**2)**1.5
    dydt81 = v_y81
    dv_ydt81 = - G * M * y81 / (x81**2 + y81**2)**1.5

    dxdt82 = v_x82
    dv_xdt82 = - G * M * x82 / (x82**2 + y82**2)**1.5
    dydt82 = v_y82
    dv_ydt82 = - G * M * y82 / (x82**2 + y82**2)**1.5

    dxdt83 = v_x83
    dv_xdt83 = - G * M * x83 / (x83**2 + y83**2)**1.5
    dydt83 = v_y83
    dv_ydt83 = - G * M * y83 / (x83**2 + y83**2)**1.5

    dxdt84 = v_x84
    dv_xdt84 = - G * M * x84 / (x84**2 + y84**2)**1.5
    dydt84 = v_y84
    dv_ydt84 = - G * M * y84 / (x84**2 + y84**2)**1.5

    dxdt85 = v_x85
    dv_xdt85 = - G * M * x85 / (x85**2 + y85**2)**1.5
    dydt85 = v_y85
    dv_ydt85 = - G * M * y85 / (x85**2 + y85**2)**1.5

    dxdt86 = v_x86
    dv_xdt86 = - G * M * x86 / (x86**2 + y86**2)**1.5
    dydt86 = v_y86
    dv_ydt86 = - G * M * y86 / (x86**2 + y86**2)**1.5

    dxdt87 = v_x87
    dv_xdt87 = - G * M * x87 / (x87**2 + y87**2)**1.5
    dydt87 = v_y87
    dv_ydt87 = - G * M * y87 / (x87**2 + y87**2)**1.5

    dxdt88 = v_x88
    dv_xdt88 = - G * M * x88 / (x88**2 + y88**2)**1.5
    dydt88 = v_y88
    dv_ydt88 = - G * M * y88 / (x88**2 + y88**2)**1.5

    dxdt89 = v_x89
    dv_xdt89 = - G * M * x89 / (x89**2 + y89**2)**1.5
    dydt89 = v_y89
    dv_ydt89 = - G * M * y89 / (x89**2 + y89**2)**1.5

    dxdt90 = v_x90
    dv_xdt90 = - G * M * x90 / (x90**2 + y90**2)**1.5
    dydt90 = v_y90
    dv_ydt90 = - G * M * y90 / (x90**2 + y90**2)**1.5

    dxdt91 = v_x91
    dv_xdt91 = - G * M * x91 / (x91**2 + y91**2)**1.5
    dydt91 = v_y91
    dv_ydt91 = - G * M * y91 / (x91**2 + y91**2)**1.5

    dxdt92 = v_x92
    dv_xdt92 = - G * M * x92 / (x92**2 + y92**2)**1.5
    dydt92 = v_y92
    dv_ydt92 = - G * M * y92 / (x92**2 + y92**2)**1.5

    dxdt93 = v_x93
    dv_xdt93 = - G * M * x93 / (x93**2 + y93**2)**1.5
    dydt93 = v_y93
    dv_ydt93 = - G * M * y93 / (x93**2 + y93**2)**1.5

    dxdt94 = v_x94
    dv_xdt94 = - G * M * x94 / (x94**2 + y94**2)**1.5
    dydt94 = v_y94
    dv_ydt94 = - G * M * y94 / (x94**2 + y94**2)**1.5

    dxdt95 = v_x95
    dv_xdt95 = - G * M * x95 / (x95**2 + y95**2)**1.5
    dydt95 = v_y95
    dv_ydt95 = - G * M * y95 / (x95**2 + y95**2)**1.5

    dxdt96 = v_x96
    dv_xdt96 = - G * M * x96 / (x96**2 + y96**2)**1.5
    dydt96 = v_y96
    dv_ydt96 = - G * M * y96 / (x96**2 + y96**2)**1.5

    dxdt97 = v_x97
    dv_xdt97 = - G * M * x97 / (x97**2 + y97**2)**1.5
    dydt97 = v_y97
    dv_ydt97 = - G * M * y97 / (x97**2 + y97**2)**1.5

    dxdt98 = v_x98
    dv_xdt98 = - G * M * x98 / (x98**2 + y98**2)**1.5
    dydt98 = v_y98
    dv_ydt98 = - G * M * y98 / (x98**2 + y98**2)**1.5

    dxdt99 = v_x99
    dv_xdt99 = - G * M * x99 / (x99**2 + y99**2)**1.5
    dydt99 = v_y99
    dv_ydt99 = - G * M * y99 / (x99**2 + y99**2)**1.5

    dxdt100 = v_x100
    dv_xdt100 = - G * M * x100 / (x100**2 + y100**2)**1.5
    dydt100 = v_y100
    dv_ydt100 = - G * M * y100 / (x100**2 + y100**2)**1.5

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
            dxdt100, dv_xdt100, dydt100, dv_ydt100)

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
v_y01 = Ve(0.5*ae, ae)

x02 = 0.5 * ae *  np.cos(np.deg2rad(3.6))
v_x02 = - Ve(0.5*ae, ae) * np.sin(np.deg2rad(3.6))
y02 = - 0.5 * ae *  np.sin(np.deg2rad(3.6))
v_y02 = Ve(0.5*ae, ae) * np.cos(np.deg2rad(3.6))

x03 = 0.5 * ae *  np.cos(np.deg2rad(7.2))
v_x03 = - Ve(0.5*ae, ae) * np.sin(np.deg2rad(7.2))
y03 = - 0.5 * ae *  np.sin(np.deg2rad(7.2))
v_y03 = Ve(0.5*ae, ae) * np.cos(np.deg2rad(7.2))

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
