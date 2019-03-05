# -- coding: utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import signal
import os
def file_name(file_dir):
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        if files==[]:
            continue
        for file in files:
            if file.endswith("tsv"):
                file_names.append(root+"/"+file)  # 当前目录路径
    return file_names
file_name_all = file_name('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/DATA/DATA')
for one_file_name in file_name_all:
    file_name_detail = one_file_name.split('/')
    userid = file_name_detail[9]
    seqid = file_name_detail[10]
    saved_path = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/extract_data/"+userid+"/"+seqid
    if not os.path.isdir(saved_path):
        os.makedirs(saved_path)
        print("creat path : " + saved_path)
    else:
        print("pass: " + saved_path)
        continue
    train = pd.read_csv(one_file_name, sep='\t')
    train = train.values

    train_mean = np.mean(train, axis=0)
    train = train - train_mean
    sample_num = np.size(train, 0)
    t = np.arange(0, sample_num, 1)
    #: (0) sensor-1 x, (1) sensor-1 y,(2) sensor-1 z, (4) sensor-2 x, (5) sensor-2 y, (6) sensor-2 z
    # 手背是sensor-1，手臂是sensor-2
    # z-axis pointing away from the skin surface, while y-axis pointing towards the fingers.
    # 数据集中标的左右指的是病人的生理上的左右手，而不是图像中的左右
    for clo in [0,1,2,4,5,6]:
        plt.clf()
        acc_data = train[:, clo]
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.set_title('1_x')
        ax1.set_xlabel('gamma-value')
        ax1.set_ylabel('R-value')
        ax1.plot(t, acc_data, c='k', linewidth = '1')

        b, a = signal.butter(8, 0.0062, 'highpass') #  wn=2*2/1000=0.004
        acc_data = signal.filtfilt(b, a, acc_data)
        axf = fig.add_subplot(4, 1, 2)
        axf.set_title('1_x_s')
        axf.set_xlabel('gamma-value')
        axf.set_ylabel('R-value')
        axf.plot(t, acc_data, c='k',linewidth = '1')

        ax1_v = fig.add_subplot(4, 1, 3)
        ax1_v.set_title('1_x_v')
        ax1_v.set_xlabel('gamma-value')
        ax1_v.set_ylabel('R-value')
        acc_data_v = np.zeros(sample_num)
        for i in range(1, sample_num+1):
            if i >=1000:
                acc_data_v[i - 1] = simps(acc_data[i-1000:i], t[i-1000:i])
            else:
                acc_data_v[i-1] = simps(acc_data[:i], t[:i])
        ax1_v.plot(t, acc_data_v, c='k',linewidth = '1')

        ax1_vs = fig.add_subplot(4, 1, 4)
        ax1_vs.set_title('1_x_v')
        ax1_vs.set_xlabel('gamma-value')
        ax1_vs.set_ylabel('R-value')
        acc_data_vs = np.zeros(sample_num)
        for i in range(1, sample_num+1):
            if i >=1000:
                acc_data_vs[i - 1] = simps(acc_data_v[i-1000:i], t[i-1000:i])
            else:
                acc_data_vs[i-1] = simps(acc_data_v[:i], t[:i])
        acc_data_vs = signal.filtfilt(b, a, acc_data_vs)

        acc_data_vs_mean = np.mean(acc_data_vs)
        acc_data_vs = acc_data_vs - acc_data_vs_mean

        ax1_vs.plot(t, acc_data_vs, c='k', linewidth = '1')

        np.savetxt(saved_path+"/" + str(clo).zfill(2)+ ".txt", acc_data_vs)  # 缺省按照'%.18e'格式保存数据，以空格分隔
        plt.savefig(saved_path+"/" + str(clo).zfill(2) + ".png")
