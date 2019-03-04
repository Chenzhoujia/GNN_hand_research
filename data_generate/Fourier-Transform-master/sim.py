# a1,a2,b1,b2
# Gaussian functions
#coding=utf-8
import matplotlib.pyplot as plt
import math,random
import numpy as np
from scipy.integrate import simps
sample_num = 1000
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def compute_coefﬁcients(A,B,C):
    return A * random.gauss(B, C)

def compute_fre():
    w = 4.5 + 0.75*sigmoid(random.uniform(-1e2, 1e2))
    return w

def fre_randomwalk(w, k):
    for i in range(k):
        w = w + random.gauss(0, 1e-7 ** 0.5)
    return w

def truncated_Fourier_model():
    # compute_coefﬁcients
    a_A = [4, 1]       #高度
    a_B = [1.25, 1.25]#中心的坐标
    a_C = [0.59, 0.59]#标准方差

    b_A = [2, 3.5]
    b_B = [1.25, 1.25]
    b_C = [0.63, 0.63]
    k = len(a_A)

    result = np.zeros(sample_num)

    for u in range(sample_num):
        if u%1000==0:
            # 参数
            an_k = []
            bn_k = []
            fre = []
            fre.append(compute_fre())
            for n in range(k):
                an_k.append(compute_coefﬁcients(a_A[n], a_B[n], a_C[n]))
                bn_k.append(compute_coefﬁcients(b_A[n], b_B[n], b_C[n]))
                if n != 0:
                    fre.append(fre_randomwalk(fre[n - 1], n))
            # 参数 end
        z = 0
        for k_i in range(1, k + 1):  # 1,2
            p = 2 * np.pi * u * fre[k_i-1] * k_i / sample_num
            z += bn_k[k_i-1]*np.cos(p) * 1j + an_k[k_i-1]*np.sin(p)
        result[u] = (z / np.sqrt(n)).real
    #result += random.gauss(0, 1e-4 ** 0.5)
    return result
St = truncated_Fourier_model()
t = np.arange(0, sample_num, 1)
x = np.linspace(0, np.size(St[0]), np.size(St[0]))
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax1.set_title('acc')
ax1.set_xlabel('gamma-value')
ax1.set_ylabel('R-value')
ax1.plot(t, St, c='k',linewidth = '1')


ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title('v')
ax2.set_xlabel('gamma-value')
ax2.set_ylabel('R-value')
St_v = np.zeros(sample_num)
St_v_sum=0
for i in range(1, sample_num+1):
    St_v[i-1] = simps(St[:i], t[:i])
    St_v_sum += St_v[i-1]
St_v_sum = St_v_sum/sample_num
St_v = St_v-St_v_sum
ax2.plot(t, St_v, c='k',linewidth = '1')

ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('s')
ax3.set_xlabel('gamma-value')
ax3.set_ylabel('R-value')
St_s = np.zeros(sample_num)
for i in range(1, sample_num+1):
    St_s[i-1] = simps(St_v[:i], t[:i])
ax3.plot(t, St_s, c='k',linewidth = '1')

plt.show()

