#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import math

a = [0, 1, 3, 3, 2, 2]
b = [3, 4, 2, 1]
c = [3, 3, 2, 3]

a = np.array(a)
b = np.array(b)
c = np.array(c)


def rotate(angle, valuex, valuey):
    rotatex = math.cos(angle) * valuex - math.sin(angle) * valuey
    rotatey = math.cos(angle) * valuey + math.sin(angle) * valuex
    rotatex = rotatex.tolist()
    rotatey = rotatey.tolist()
    xy = rotatex + rotatey
    return xy


def getLen(x1, y1, x2, y2):
    diff_x = (x1 - x2) ** 2
    diff_y = (y1 - y2) ** 2
    length = np.sqrt(diff_x + diff_y)
    return length


lie = np.linspace(0, 2, 20)
for i in lie:
    t = math.pi * i
    a1 = rotate(t, a[0:3], a[3:6])
    b1 = rotate(t, b[0:2], b[2:4])
    c1 = rotate(t, c[0:2], c[2:4])
    len1 = getLen(a[0], a[3], a[1], a[4])
    len2 = getLen(a1[0], a1[3], a1[1], a1[4])
    print(len1)
    print(len2)
    print("旋转后长度是否相等", len1 == len2)
    plt.plot(a[0:3], a[3:6], color='green')
    plt.plot(b[0:2], b[2:4], color='green')
    plt.plot(c[0:2], c[2:4], color='green')
    plt.plot(a1[0:3], a1[3:6], color='red')
    plt.plot(b1[0:2], b1[2:4], color='red')
    plt.plot(c1[0:2], c1[2:4], color='red')
    plt.xticks(np.arange(-5, 5, 0.5))
    plt.yticks(np.arange(-5, 5, 0.5))

    plt.show()