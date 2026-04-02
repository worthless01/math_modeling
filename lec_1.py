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
ae = 149 * 10**9
edge = 2.5 * ae
e = 0.5

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
kappa2 = 0.6
x02 = kappa2 * ae *  np.cos(np.deg2rad(alpha2))
v_x02 = - Ve(1, kappa2*ae) * np.sin(np.deg2rad(alpha2))
y02 = kappa2 * ae *  np.sin(np.deg2rad(alpha2))
v_y02 = Ve(1, kappa2*ae) * np.cos(np.deg2rad(alpha2))

alpha3 = 80
kappa3 = 0.7
x03 = kappa3 * ae *  np.cos(np.deg2rad(alpha3))
v_x03 = - Ve(1, kappa3*ae) * np.sin(np.deg2rad(alpha3))
y03 = kappa3 * ae *  np.sin(np.deg2rad(alpha3))
v_y03 = Ve(1, kappa3*ae) * np.cos(np.deg2rad(alpha3))

alpha4 = 85
kappa4 = 0.8
x04 = kappa4 * ae * np.cos(np.deg2rad(alpha4))
v_x04 = - Ve(1, kappa4*ae) * np.sin(np.deg2rad(alpha4))
y04 = kappa4 * ae * np.sin(np.deg2rad(alpha4))
v_y04 = Ve(1, kappa4*ae) * np.cos(np.deg2rad(alpha4))

alpha5 = 90
kappa5 = 0.9
x05 = kappa5 * ae * np.cos(np.deg2rad(alpha5))
v_x05 = - Ve(1, kappa5*ae) * np.sin(np.deg2rad(alpha5))
y05 = kappa5 * ae * np.sin(np.deg2rad(alpha5))
v_y05 = Ve(1, kappa5*ae) * np.cos(np.deg2rad(alpha5))

alpha6 = 95
kappa6 = 1
x06 = kappa6 * ae * np.cos(np.deg2rad(alpha6))
v_x06 = - Ve(1, kappa6*ae) * np.sin(np.deg2rad(alpha6))
y06 = kappa6 * ae * np.sin(np.deg2rad(alpha6))
v_y06 = Ve(1, kappa6*ae) * np.cos(np.deg2rad(alpha6))

alpha7 = 100
kappa7 = 1.1
x07 = kappa7 * ae * np.cos(np.deg2rad(alpha7))
v_x07 = - Ve(1, kappa7*ae) * np.sin(np.deg2rad(alpha7))
y07 = kappa7 * ae * np.sin(np.deg2rad(alpha7))
v_y07 = Ve(1, kappa7*ae) * np.cos(np.deg2rad(alpha7))

alpha8 = 105
kappa8 = 1.2
x08 = kappa8 * ae * np.cos(np.deg2rad(alpha8))
v_x08 = - Ve(1, kappa8*ae) * np.sin(np.deg2rad(alpha8))
y08 = kappa8 * ae * np.sin(np.deg2rad(alpha8))
v_y08 = Ve(1, kappa8*ae) * np.cos(np.deg2rad(alpha8))

alpha9 = 110
kappa9 = 1.3
x09 = kappa9 * ae * np.cos(np.deg2rad(alpha9))
v_x09 = - Ve(1, kappa9*ae) * np.sin(np.deg2rad(alpha9))
y09 = kappa9 * ae * np.sin(np.deg2rad(alpha9))
v_y09 = Ve(1, kappa9*ae) * np.cos(np.deg2rad(alpha9))

alpha10 = 115
kappa10 = 1.4
x010 = kappa10 * ae * np.cos(np.deg2rad(alpha10))
v_x010 = - Ve(1, kappa10*ae) * np.sin(np.deg2rad(alpha10))
y010 =  kappa10 * ae * np.sin(np.deg2rad(alpha10))
v_y010 = Ve(1, kappa10*ae) * np.cos(np.deg2rad(alpha10))

alpha11 = 120
kappa11 = 1.5
x011 = kappa11 * ae * np.cos(np.deg2rad(alpha11))
v_x011 = - Ve(1, kappa11*ae) * np.sin(np.deg2rad(alpha11))
y011 =  kappa11 * ae * np.sin(np.deg2rad(alpha11))
v_y011 = Ve(1, kappa11*ae) * np.cos(np.deg2rad(alpha11))

x012 = 0.522 * ae * np.cos(np.deg2rad(39.6))
v_x012 = - Ve(0.522*ae, ae) * np.sin(np.deg2rad(39.6))
y012 = - 0.522 * ae * np.sin(np.deg2rad(39.6))
v_y012 = Ve(0.522*ae, ae) * np.cos(np.deg2rad(39.6))

x013 = 0.524 * ae * np.cos(np.deg2rad(43.2))
v_x013 = - Ve(0.524*ae, ae) * np.sin(np.deg2rad(43.2))
y013 = - 0.524 * ae * np.sin(np.deg2rad(43.2))
v_y013 = Ve(0.524*ae, ae) * np.cos(np.deg2rad(43.2))

x014 = 0.526 * ae * np.cos(np.deg2rad(46.8))
v_x014 = - Ve(0.526*ae, ae) * np.sin(np.deg2rad(46.8))
y014 = - 0.526 * ae * np.sin(np.deg2rad(46.8))
v_y014 = Ve(0.526*ae, ae) * np.cos(np.deg2rad(46.8))

x015 = 0.528 * ae * np.cos(np.deg2rad(50.4))
v_x015 = - Ve(0.528*ae, ae) * np.sin(np.deg2rad(50.4))
y015 = - 0.528 * ae * np.sin(np.deg2rad(50.4))
v_y015 = Ve(0.528*ae, ae) * np.cos(np.deg2rad(50.4))

x016 = 0.53 * ae * np.cos(np.deg2rad(54.0))
v_x016 = - Ve(0.53*ae, ae) * np.sin(np.deg2rad(54.0))
y016 = - 0.53 * ae * np.sin(np.deg2rad(54.0))
v_y016 = Ve(0.53*ae, ae) * np.cos(np.deg2rad(54.0))

x017 = 0.532 * ae * np.cos(np.deg2rad(57.6))
v_x017 = - Ve(0.532*ae, ae) * np.sin(np.deg2rad(57.6))
y017 = - 0.532 * ae * np.sin(np.deg2rad(57.6))
v_y017 = Ve(0.532*ae, ae) * np.cos(np.deg2rad(57.6))

x018 = 0.534 * ae * np.cos(np.deg2rad(61.2))
v_x018 = - Ve(0.534*ae, ae) * np.sin(np.deg2rad(61.2))
y018 = - 0.534 * ae * np.sin(np.deg2rad(61.2))
v_y018 = Ve(0.534*ae, ae) * np.cos(np.deg2rad(61.2))

x019 = 0.536 * ae * np.cos(np.deg2rad(64.8))
v_x019 = - Ve(0.536*ae, ae) * np.sin(np.deg2rad(64.8))
y019 = - 0.536 * ae * np.sin(np.deg2rad(64.8))
v_y019 = Ve(0.536*ae, ae) * np.cos(np.deg2rad(64.8))

x020 = 0.538 * ae * np.cos(np.deg2rad(68.4))
v_x020 = - Ve(0.538*ae, ae) * np.sin(np.deg2rad(68.4))
y020 = - 0.538 * ae * np.sin(np.deg2rad(68.4))
v_y020 = Ve(0.538*ae, ae) * np.cos(np.deg2rad(68.4))

x021 = 0.54 * ae * np.cos(np.deg2rad(72.0))
v_x021 = - Ve(0.54*ae, ae) * np.sin(np.deg2rad(72.0))
y021 = - 0.54 * ae * np.sin(np.deg2rad(72.0))
v_y021 = Ve(0.54*ae, ae) * np.cos(np.deg2rad(72.0))

x022 = 0.542 * ae * np.cos(np.deg2rad(75.6))
v_x022 = - Ve(0.542*ae, ae) * np.sin(np.deg2rad(75.6))
y022 = - 0.542 * ae * np.sin(np.deg2rad(75.6))
v_y022 = Ve(0.542*ae, ae) * np.cos(np.deg2rad(75.6))

x023 = 0.544 * ae * np.cos(np.deg2rad(79.2))
v_x023 = - Ve(0.544*ae, ae) * np.sin(np.deg2rad(79.2))
y023 = - 0.544 * ae * np.sin(np.deg2rad(79.2))
v_y023 = Ve(0.544*ae, ae) * np.cos(np.deg2rad(79.2))

x024 = 0.546 * ae * np.cos(np.deg2rad(82.8))
v_x024 = - Ve(0.546*ae, ae) * np.sin(np.deg2rad(82.8))
y024 = - 0.546 * ae * np.sin(np.deg2rad(82.8))
v_y024 = Ve(0.546*ae, ae) * np.cos(np.deg2rad(82.8))

x025 = 0.548 * ae * np.cos(np.deg2rad(86.4))
v_x025 = - Ve(0.548*ae, ae) * np.sin(np.deg2rad(86.4))
y025 = - 0.548 * ae * np.sin(np.deg2rad(86.4))
v_y025 = Ve(0.548*ae, ae) * np.cos(np.deg2rad(86.4))

x026 = 0.55 * ae * np.cos(np.deg2rad(90.0))
v_x026 = - Ve(0.55*ae, ae) * np.sin(np.deg2rad(90.0))
y026 = - 0.55 * ae * np.sin(np.deg2rad(90.0))
v_y026 = Ve(0.55*ae, ae) * np.cos(np.deg2rad(90.0))

x027 = 0.552 * ae * np.cos(np.deg2rad(93.6))
v_x027 = - Ve(0.552*ae, ae) * np.sin(np.deg2rad(93.6))
y027 = - 0.552 * ae * np.sin(np.deg2rad(93.6))
v_y027 = Ve(0.552*ae, ae) * np.cos(np.deg2rad(93.6))

x028 = 0.554 * ae * np.cos(np.deg2rad(97.2))
v_x028 = - Ve(0.554*ae, ae) * np.sin(np.deg2rad(97.2))
y028 = - 0.554 * ae * np.sin(np.deg2rad(97.2))
v_y028 = Ve(0.554*ae, ae) * np.cos(np.deg2rad(97.2))

x029 = 0.556 * ae * np.cos(np.deg2rad(100.8))
v_x029 = - Ve(0.556*ae, ae) * np.sin(np.deg2rad(100.8))
y029 = - 0.556 * ae * np.sin(np.deg2rad(100.8))
v_y029 = Ve(0.556*ae, ae) * np.cos(np.deg2rad(100.8))

x030 = 0.558 * ae * np.cos(np.deg2rad(104.4))
v_x030 = - Ve(0.558*ae, ae) * np.sin(np.deg2rad(104.4))
y030 = - 0.558 * ae * np.sin(np.deg2rad(104.4))
v_y030 = Ve(0.558*ae, ae) * np.cos(np.deg2rad(104.4))

x031 = 0.56 * ae * np.cos(np.deg2rad(108.0))
v_x031 = - Ve(0.56*ae, ae) * np.sin(np.deg2rad(108.0))
y031 = - 0.56 * ae * np.sin(np.deg2rad(108.0))
v_y031 = Ve(0.56*ae, ae) * np.cos(np.deg2rad(108.0))

x032 = 0.562 * ae * np.cos(np.deg2rad(111.6))
v_x032 = - Ve(0.562*ae, ae) * np.sin(np.deg2rad(111.6))
y032 = - 0.562 * ae * np.sin(np.deg2rad(111.6))
v_y032 = Ve(0.562*ae, ae) * np.cos(np.deg2rad(111.6))

x033 = 0.564 * ae * np.cos(np.deg2rad(115.2))
v_x033 = - Ve(0.564*ae, ae) * np.sin(np.deg2rad(115.2))
y033 = - 0.564 * ae * np.sin(np.deg2rad(115.2))
v_y033 = Ve(0.564*ae, ae) * np.cos(np.deg2rad(115.2))

x034 = 0.566 * ae * np.cos(np.deg2rad(118.8))
v_x034 = - Ve(0.566*ae, ae) * np.sin(np.deg2rad(118.8))
y034 = - 0.566 * ae * np.sin(np.deg2rad(118.8))
v_y034 = Ve(0.566*ae, ae) * np.cos(np.deg2rad(118.8))

x035 = 0.568 * ae * np.cos(np.deg2rad(122.4))
v_x035 = - Ve(0.568*ae, ae) * np.sin(np.deg2rad(122.4))
y035 = - 0.568 * ae * np.sin(np.deg2rad(122.4))
v_y035 = Ve(0.568*ae, ae) * np.cos(np.deg2rad(122.4))

x036 = 0.57 * ae * np.cos(np.deg2rad(126.0))
v_x036 = - Ve(0.57*ae, ae) * np.sin(np.deg2rad(126.0))
y036 = - 0.57 * ae * np.sin(np.deg2rad(126.0))
v_y036 = Ve(0.57*ae, ae) * np.cos(np.deg2rad(126.0))

x037 = 0.572 * ae * np.cos(np.deg2rad(129.6))
v_x037 = - Ve(0.572*ae, ae) * np.sin(np.deg2rad(129.6))
y037 = - 0.572 * ae * np.sin(np.deg2rad(129.6))
v_y037 = Ve(0.572*ae, ae) * np.cos(np.deg2rad(129.6))

x038 = 0.574 * ae * np.cos(np.deg2rad(133.2))
v_x038 = - Ve(0.574*ae, ae) * np.sin(np.deg2rad(133.2))
y038 = - 0.574 * ae * np.sin(np.deg2rad(133.2))
v_y038 = Ve(0.574*ae, ae) * np.cos(np.deg2rad(133.2))

x039 = 0.576 * ae * np.cos(np.deg2rad(136.8))
v_x039 = - Ve(0.576*ae, ae) * np.sin(np.deg2rad(136.8))
y039 = - 0.576 * ae * np.sin(np.deg2rad(136.8))
v_y039 = Ve(0.576*ae, ae) * np.cos(np.deg2rad(136.8))

x040 = 0.578 * ae * np.cos(np.deg2rad(140.4))
v_x040 = - Ve(0.578*ae, ae) * np.sin(np.deg2rad(140.4))
y040 = - 0.578 * ae * np.sin(np.deg2rad(140.4))
v_y040 = Ve(0.578*ae, ae) * np.cos(np.deg2rad(140.4))

x041 = 0.58 * ae * np.cos(np.deg2rad(144.0))
v_x041 = - Ve(0.58*ae, ae) * np.sin(np.deg2rad(144.0))
y041 = - 0.58 * ae * np.sin(np.deg2rad(144.0))
v_y041 = Ve(0.58*ae, ae) * np.cos(np.deg2rad(144.0))

x042 = 0.582 * ae * np.cos(np.deg2rad(147.6))
v_x042 = - Ve(0.582*ae, ae) * np.sin(np.deg2rad(147.6))
y042 = - 0.582 * ae * np.sin(np.deg2rad(147.6))
v_y042 = Ve(0.582*ae, ae) * np.cos(np.deg2rad(147.6))

x043 = 0.584 * ae * np.cos(np.deg2rad(151.2))
v_x043 = - Ve(0.548*ae, ae) * np.sin(np.deg2rad(151.2))
y043 = - 0.584 * ae * np.sin(np.deg2rad(151.2))
v_y043 = Ve(0.584*ae, ae) * np.cos(np.deg2rad(151.2))

x044 = 0.586 * ae * np.cos(np.deg2rad(154.8))
v_x044 = - Ve(0.586*ae, ae) * np.sin(np.deg2rad(154.8))
y044 = - 0.586 * ae * np.sin(np.deg2rad(154.8))
v_y044 = Ve(0.568*ae, ae) * np.cos(np.deg2rad(154.8))

x045 = 0.59 * ae * np.cos(np.deg2rad(158.4))
v_x045 = - Ve(0.59*ae, ae) * np.sin(np.deg2rad(158.4))
y045 = - 0.59 * ae * np.sin(np.deg2rad(158.4))
v_y045 = Ve(0.59*ae, ae) * np.cos(np.deg2rad(158.4))

x046 = 0.592 * ae * np.cos(np.deg2rad(162.0))
v_x046 = - Ve(0.592*ae, ae) * np.sin(np.deg2rad(162.0))
y046 = - 0.592 * ae * np.sin(np.deg2rad(162.0))
v_y046 = Ve(0.592*ae, ae) * np.cos(np.deg2rad(162.0))

x047 = 0.592 * ae * np.cos(np.deg2rad(165.6))
v_x047 = - Ve(0.592*ae, ae) * np.sin(np.deg2rad(165.6))
y047 = - 0.592 * ae * np.sin(np.deg2rad(165.6))
v_y047 = Ve(0.592*ae, ae) * np.cos(np.deg2rad(165.6))

x048 = 0.594 * ae * np.cos(np.deg2rad(169.2))
v_x048 = - Ve(0.594*ae, ae) * np.sin(np.deg2rad(169.2))
y048 = - 0.594 * ae * np.sin(np.deg2rad(169.2))
v_y048 = Ve(0.594*ae, ae) * np.cos(np.deg2rad(169.2))

x049 = 0.596 * ae * np.cos(np.deg2rad(172.8))
v_x049 = - Ve(0.596*ae, ae) * np.sin(np.deg2rad(172.8))
y049 = - 0.596 * ae * np.sin(np.deg2rad(172.8))
v_y049 = Ve(0.596*ae, ae) * np.cos(np.deg2rad(172.8))

x050 = 0.598 * ae * np.cos(np.deg2rad(176.4))
v_x050 = - Ve(0.598*ae, ae) * np.sin(np.deg2rad(176.4))
y050 = - 0.598 * ae * np.sin(np.deg2rad(176.4))
v_y050 = Ve(0.598*ae, ae) * np.cos(np.deg2rad(176.4))

x051 = 0.6 * ae * np.cos(np.deg2rad(180.0))
v_x051 = - Ve(0.6*ae, ae) * np.sin(np.deg2rad(180.0))
y051 = - 0.6 * ae * np.sin(np.deg2rad(180.0))
v_y051 = Ve(0.6*ae, ae) * np.cos(np.deg2rad(180.0))

x052 = 0.602 * ae * np.cos(np.deg2rad(183.6))
v_x052 = - Ve(0.602*ae, ae) * np.sin(np.deg2rad(183.6))
y052 = - 0.602 * ae * np.sin(np.deg2rad(183.6))
v_y052 = Ve(0.602*ae, ae) * np.cos(np.deg2rad(183.6))

x053 = 0.604 * ae * np.cos(np.deg2rad(187.2))
v_x053 = - Ve(0.604*ae, ae) * np.sin(np.deg2rad(187.2))
y053 = - 0.604 * ae * np.sin(np.deg2rad(187.2))
v_y053 = Ve(0.604*ae, ae) * np.cos(np.deg2rad(187.2))

x054 = 0.606 * ae * np.cos(np.deg2rad(190.8))
v_x054 = - Ve(0.606*ae, ae) * np.sin(np.deg2rad(190.8))
y054 = - 0.606 * ae * np.sin(np.deg2rad(190.8))
v_y054 = Ve(0.606*ae, ae) * np.cos(np.deg2rad(190.8))

x055 = 0.608 * ae * np.cos(np.deg2rad(194.4))
v_x055 = - Ve(0.608*ae, ae) * np.sin(np.deg2rad(194.4))
y055 = - 0.608 * ae * np.sin(np.deg2rad(194.4))
v_y055 = Ve(0.608*ae, ae) * np.cos(np.deg2rad(194.4))

x056 = 0.61 * ae * np.cos(np.deg2rad(198.0))
v_x056 = - Ve(0.61*ae, ae) * np.sin(np.deg2rad(198.0))
y056 = - 0.61 * ae * np.sin(np.deg2rad(198.0))
v_y056 = Ve(0.61*ae, ae) * np.cos(np.deg2rad(198.0))

x057 = 0.612 * ae * np.cos(np.deg2rad(201.6))
v_x057 = - Ve(0.612*ae, ae) * np.sin(np.deg2rad(201.6))
y057 = - 0.612 * ae * np.sin(np.deg2rad(201.6))
v_y057 = Ve(0.612*ae, ae) * np.cos(np.deg2rad(201.6))

x058 = 0.612 * ae * np.cos(np.deg2rad(205.2))
v_x058 = - Ve(0.612*ae, ae) * np.sin(np.deg2rad(205.2))
y058 = - 0.612 * ae * np.sin(np.deg2rad(205.2))
v_y058 = Ve(0.612*ae, ae) * np.cos(np.deg2rad(205.2))

x059 = 0.614 * ae * np.cos(np.deg2rad(208.8))
v_x059 = - Ve(0.614*ae, ae) * np.sin(np.deg2rad(208.8))
y059 = - 0.614 * ae * np.sin(np.deg2rad(208.8))
v_y059 = Ve(0.614*ae, ae) * np.cos(np.deg2rad(208.8))

x060 = 0.616 * ae * np.cos(np.deg2rad(212.4))
v_x060 = - Ve(0.616*ae, ae) * np.sin(np.deg2rad(212.4))
y060 = - 0.616 * ae * np.sin(np.deg2rad(212.4))
v_y060 = Ve(0.616*ae, ae) * np.cos(np.deg2rad(212.4))

x061 = 0.618 * ae * np.cos(np.deg2rad(216.0))
v_x061 = - Ve(0.618*ae, ae) * np.sin(np.deg2rad(216.0))
y061 = - 0.618 * ae * np.sin(np.deg2rad(216.0))
v_y061 = Ve(0.618*ae, ae) * np.cos(np.deg2rad(216.0))

x062 = 0.62 * ae * np.cos(np.deg2rad(219.6))
v_x062 = - Ve(0.62*ae, ae) * np.sin(np.deg2rad(219.6))
y062 = - 0.62 * ae * np.sin(np.deg2rad(219.6))
v_y062 = Ve(0.62*ae, ae) * np.cos(np.deg2rad(219.6))

x063 = 0.622 * ae * np.cos(np.deg2rad(223.2))
v_x063 = - Ve(0.622*ae, ae) * np.sin(np.deg2rad(223.2))
y063 = - 0.622 * ae * np.sin(np.deg2rad(223.2))
v_y063 = Ve(0.622*ae, ae) * np.cos(np.deg2rad(223.2))

x064 = 0.624 * ae * np.cos(np.deg2rad(226.8))
v_x064 = - Ve(0.624*ae, ae) * np.sin(np.deg2rad(226.8))
y064 = - 0.624 * ae * np.sin(np.deg2rad(226.8))
v_y064 = Ve(0.624*ae, ae) * np.cos(np.deg2rad(226.8))

x065 = 0.626 * ae * np.cos(np.deg2rad(230.4))
v_x065 = - Ve(0.626*ae, ae) * np.sin(np.deg2rad(230.4))
y065 = - 0.626 * ae * np.sin(np.deg2rad(230.4))
v_y065 = Ve(0.626*ae, ae) * np.cos(np.deg2rad(230.4))

x066 = 0.628 * ae * np.cos(np.deg2rad(234.0))
v_x066 = - Ve(0.628*ae, ae) * np.sin(np.deg2rad(234.0))
y066 = - 0.628 * ae * np.sin(np.deg2rad(234.0))
v_y066 = Ve(0.628*ae, ae) * np.cos(np.deg2rad(234.0))

x067 = 0.63 * ae * np.cos(np.deg2rad(237.6))
v_x067 = - Ve(0.63*ae, ae) * np.sin(np.deg2rad(237.6))
y067 = - 0.63 * ae * np.sin(np.deg2rad(237.6))
v_y067 = Ve(0.63*ae, ae) * np.cos(np.deg2rad(237.6))

x068 = 0.632 * ae * np.cos(np.deg2rad(241.2))
v_x068 = - Ve(0.632*ae, ae) * np.sin(np.deg2rad(241.2))
y068 = - 0.632 * ae * np.sin(np.deg2rad(241.2))
v_y068 = Ve(0.632*ae, ae) * np.cos(np.deg2rad(241.2))

x069 = 0.634 * ae * np.cos(np.deg2rad(244.8))
v_x069 = - Ve(0.634*ae, ae) * np.sin(np.deg2rad(244.8))
y069 = - 0.634 * ae * np.sin(np.deg2rad(244.8))
v_y069 = Ve(0.634*ae, ae) * np.cos(np.deg2rad(244.8))

x070 = 0.636 * ae * np.cos(np.deg2rad(248.4))
v_x070 = - Ve(0.636*ae, ae) * np.sin(np.deg2rad(248.4))
y070 = - 0.636 * ae * np.sin(np.deg2rad(248.4))
v_y070 = Ve(0.636*ae, ae) * np.cos(np.deg2rad(248.4))

x071 = 0.638 * ae * np.cos(np.deg2rad(252.0))
v_x071 = - Ve(0.638*ae, ae) * np.sin(np.deg2rad(252.0))
y071 = - 0.638 * ae * np.sin(np.deg2rad(252.0))
v_y071 = Ve(0.638*ae, ae) * np.cos(np.deg2rad(252.0))

x072 = 0.64 * ae * np.cos(np.deg2rad(255.6))
v_x072 = - Ve(0.64 *ae, ae) * np.sin(np.deg2rad(255.6))
y072 = - 0.64  * ae * np.sin(np.deg2rad(255.6))
v_y072 = Ve(0.64 *ae, ae) * np.cos(np.deg2rad(255.6))

x073 = 0.642 * ae * np.cos(np.deg2rad(259.2))
v_x073 = - Ve(0.642*ae, ae) * np.sin(np.deg2rad(259.2))
y073 = - 0.642 * ae * np.sin(np.deg2rad(259.2))
v_y073 = Ve(0.642*ae, ae) * np.cos(np.deg2rad(259.2))

x074 = 0.644 * ae * np.cos(np.deg2rad(262.8))
v_x074 = - Ve(0.644*ae, ae) * np.sin(np.deg2rad(262.8))
y074 = - 0.644* ae * np.sin(np.deg2rad(262.8))
v_y074 = Ve(0.644*ae, ae) * np.cos(np.deg2rad(262.8))

x075 = 0.646 * ae * np.cos(np.deg2rad(266.4))
v_x075 = - Ve(0.646*ae, ae) * np.sin(np.deg2rad(266.4))
y075 = - 0.646 * ae * np.sin(np.deg2rad(266.4))
v_y075 = Ve(0.646*ae, ae) * np.cos(np.deg2rad(266.4))

x076 = 0.648 * ae * np.cos(np.deg2rad(270.0))
v_x076 = - Ve(0.648*ae, ae) * np.sin(np.deg2rad(270.0))
y076 = - 0.648 * ae * np.sin(np.deg2rad(270.0))
v_y076 = Ve(0.648*ae, ae) * np.cos(np.deg2rad(270.0))

x077 = 0.65 * ae * np.cos(np.deg2rad(273.6))
v_x077 = - Ve(0.65*ae, ae) * np.sin(np.deg2rad(273.6))
y077 = - 0.65 * ae * np.sin(np.deg2rad(273.6))
v_y077 = Ve(0.65*ae, ae) * np.cos(np.deg2rad(273.6))

x078 = 0.652 * ae * np.cos(np.deg2rad(277.2))
v_x078 = - Ve(0.652*ae, ae) * np.sin(np.deg2rad(277.2))
y078 = - 0.652* ae * np.sin(np.deg2rad(277.2))
v_y078 = Ve(0.652*ae, ae) * np.cos(np.deg2rad(277.2))

x079 = 0.654 * ae * np.cos(np.deg2rad(280.8))
v_x079 = - Ve(0.654*ae, ae) * np.sin(np.deg2rad(280.8))
y079 = - 0.654 * ae * np.sin(np.deg2rad(280.8))
v_y079 = Ve(0.654*ae, ae) * np.cos(np.deg2rad(280.8))

x080 = 0.656 * ae * np.cos(np.deg2rad(284.4))
v_x080 = - Ve(0.656*ae, ae) * np.sin(np.deg2rad(284.4))
y080 = - 0.656 * ae * np.sin(np.deg2rad(284.4))
v_y080 = Ve(0.656*ae, ae) * np.cos(np.deg2rad(284.4))

x081 = 0.658 * ae * np.cos(np.deg2rad(288.0))
v_x081 = - Ve(0.658*ae, ae) * np.sin(np.deg2rad(288.0))
y081 = - 0.658 * ae * np.sin(np.deg2rad(288.0))
v_y081 = Ve(0.658*ae, ae) * np.cos(np.deg2rad(288.0))

x082 = 0.66 * ae * np.cos(np.deg2rad(291.6))
v_x082 = - Ve(0.66*ae, ae) * np.sin(np.deg2rad(291.6))
y082 = - 0.66 * ae * np.sin(np.deg2rad(291.6))
v_y082 = Ve(0.66*ae, ae) * np.cos(np.deg2rad(291.6))

x083 = 0.662 * ae * np.cos(np.deg2rad(295.2))
v_x083 = - Ve(0.662*ae, ae) * np.sin(np.deg2rad(295.2))
y083 = - 0.662 * ae * np.sin(np.deg2rad(295.2))
v_y083 = Ve(0.662*ae, ae) * np.cos(np.deg2rad(295.2))

x084 = 0.664 * ae * np.cos(np.deg2rad(298.8))
v_x084 = - Ve(0.664*ae, ae) * np.sin(np.deg2rad(298.8))
y084 = - 0.664 * ae * np.sin(np.deg2rad(298.8))
v_y084 = Ve(0.664*ae, ae) * np.cos(np.deg2rad(298.8))

x085 = 0.666 * ae * np.cos(np.deg2rad(302.4))
v_x085 = - Ve(0.666*ae, ae) * np.sin(np.deg2rad(302.4))
y085 = - 0.666 * ae * np.sin(np.deg2rad(302.4))
v_y085 = Ve(0.666*ae, ae) * np.cos(np.deg2rad(302.4))

x086 = 0.668 * ae * np.cos(np.deg2rad(306.0))
v_x086 = - Ve(0.668*ae, ae) * np.sin(np.deg2rad(306.0))
y086 = - 0.668 * ae * np.sin(np.deg2rad(306.0))
v_y086 = Ve(0.668*ae, ae) * np.cos(np.deg2rad(306.0))

x087 = 0.67 * ae * np.cos(np.deg2rad(309.6))
v_x087 = - Ve(0.67*ae, ae) * np.sin(np.deg2rad(309.6))
y087 = - 0.67 * ae * np.sin(np.deg2rad(309.6))
v_y087 = Ve(0.67*ae, ae) * np.cos(np.deg2rad(309.6))

x088 = 0.672 * ae * np.cos(np.deg2rad(313.2))
v_x088 = - Ve(0.672*ae, ae) * np.sin(np.deg2rad(313.2))
y088 = - 0.672 * ae * np.sin(np.deg2rad(313.2))
v_y088 = Ve(0.672*ae, ae) * np.cos(np.deg2rad(313.2))

x089 = 0.674 * ae * np.cos(np.deg2rad(316.8))
v_x089 = - Ve(0.674*ae, ae) * np.sin(np.deg2rad(316.8))
y089 = - 0.674 * ae * np.sin(np.deg2rad(316.8))
v_y089 = Ve(0.674*ae, ae) * np.cos(np.deg2rad(316.8))

x090 = 0.676 * ae * np.cos(np.deg2rad(320.4))
v_x090 = - Ve(0.676*ae, ae) * np.sin(np.deg2rad(320.4))
y090 = - 0.676 * ae * np.sin(np.deg2rad(320.4))
v_y090 = Ve(0.676*ae, ae) * np.cos(np.deg2rad(320.4))

x091 = 0.678 * ae * np.cos(np.deg2rad(324.0))
v_x091 = - Ve(0.678*ae, ae) * np.sin(np.deg2rad(324.0))
y091 = - 0.678 * ae * np.sin(np.deg2rad(324.0))
v_y091 = Ve(0.678*ae, ae) * np.cos(np.deg2rad(324.0))

x092 = 0.68 * ae * np.cos(np.deg2rad(327.6))
v_x092 = - Ve(0.68*ae, ae) * np.sin(np.deg2rad(327.6))
y092 = - 0.68 * ae * np.sin(np.deg2rad(327.6))
v_y092 = Ve(0.68*ae, ae) * np.cos(np.deg2rad(327.6))

x093 = 0.682 * ae * np.cos(np.deg2rad(331.2))
v_x093 = - Ve(0.682*ae, ae) * np.sin(np.deg2rad(331.2))
y093 = - 0.682 * ae * np.sin(np.deg2rad(331.2))
v_y093 = Ve(0.682*ae, ae) * np.cos(np.deg2rad(331.2))

x094 = 0.684 * ae * np.cos(np.deg2rad(334.8))
v_x094 = - Ve(0.684*ae, ae) * np.sin(np.deg2rad(334.8))
y094 = - 0.684 * ae * np.sin(np.deg2rad(334.8))
v_y094 = Ve(0.684*ae, ae) * np.cos(np.deg2rad(334.8))

x095 = 0.686 * ae * np.cos(np.deg2rad(338.4))
v_x095 = - Ve(0.686*ae, ae) * np.sin(np.deg2rad(338.4))
y095 = - 0.686 * ae * np.sin(np.deg2rad(338.4))
v_y095 = Ve(0.686*ae, ae) * np.cos(np.deg2rad(338.4))

x096 = 0.688 * ae * np.cos(np.deg2rad(342.0))
v_x096 = - Ve(0.688 *ae, ae) * np.sin(np.deg2rad(342.0))
y096 = - 0.688  * ae * np.sin(np.deg2rad(342.0))
v_y096 = Ve(0.688 *ae, ae) * np.cos(np.deg2rad(342.0))

x097 = 0.69 * ae * np.cos(np.deg2rad(345.6))
v_x097 = - Ve(0.69*ae, ae) * np.sin(np.deg2rad(345.6))
y097 = - 0.69 * ae * np.sin(np.deg2rad(345.6))
v_y097 = Ve(0.69*ae, ae) * np.cos(np.deg2rad(345.6))

x098 = 0.692 * ae * np.cos(np.deg2rad(349.2))
v_x098 = - Ve(0.692*ae, ae) * np.sin(np.deg2rad(349.2))
y098 = - 0.692 * ae * np.sin(np.deg2rad(349.2))
v_y098 = Ve(0.692*ae, ae) * np.cos(np.deg2rad(349.2))

x099 = 0.694 * ae * np.cos(np.deg2rad(352.8))
v_x099 = - Ve(0.694*ae, ae) * np.sin(np.deg2rad(352.8))
y099 = - 0.694 * ae * np.sin(np.deg2rad(352.8))
v_y099 = Ve(0.694*ae, ae) * np.cos(np.deg2rad(352.8))

x0100 = 0.696 * ae * np.cos(np.deg2rad(356.4))
v_x0100 = - Ve(0.696*ae, ae) * np.sin(np.deg2rad(356.4))
y0100 = - 0.696 * ae * np.sin(np.deg2rad(356.4))
v_y0100 = Ve(0.696*ae, ae) * np.cos(np.deg2rad(356.4))

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
      x0100, v_x0100, y0100, v_y0100)

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
