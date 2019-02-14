# -- coding: utf-8 --
import numpy as np
import os
from util.visual import figure_joint_skeleton
#filename = '/home/chen/Documents/.git/Mocap_SIG18_Data/training_data/User1/capture1/sequence.1_training_dense_left.trc'
def read_file(filename):
    pos = []
    with open(filename, 'r') as file_to_read:
        for i in range(5):
            lines = file_to_read.readline()
        while True:
            lines = file_to_read.readline()
            lines = lines.split('\t')
            lines = lines[2:-1]
            if len(lines)!=57:
                break
                pass
            p_tmp = [float(i) for i in lines]
            pos.append(p_tmp)
            pass
        pos = np.array(pos)
        return pos


def file_name(file_dir):
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        if files==[]:
            continue
        for file in files:
            file_names.append(root+"/"+file)  # 当前目录路径
    return file_names
        #print('sub_dirs:', dirs)  # 当前路径下所有子目录
        #print('files:', files)  # 当前路径下所有非目录子文件


file_name_all = file_name('/home/chen/Documents/.git/Mocap_SIG18_Data/training_data')
for file_name in file_name_all:
    onefile_data = read_file(file_name)
    file_name_detail = file_name.split('/')
    userid = file_name_detail[7]
    capid = file_name_detail[8]
    seqid = file_name_detail[9]
    pose_num = 0
    for pose_id in onefile_data:
        pose_num = pose_num+1
        figure_joint_skeleton(pose_id, "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake/"
                              +userid+"/"+capid+"/"+seqid+"/", pose_num)

"""
receive = []
sender = []

for i in range(19):
    for j in range(19):
        receive.append(i)
        sender.append(j)
print(sender)
print('\n')
print(receive)


a = np.arange(0, 10, 1)
b = np.arange(10, 20, 1)
for i in range(10):
    print(a, b)
    # result:[0 1 2 3 4 5 6 7 8 9] [10 11 12 13 14 15 16 17 18 19]
    state = np.random.get_state()
    np.random.shuffle(a)
    print(a)
    # result:[6 4 5 3 7 2 0 1 8 9]
    np.random.set_state(state)
    np.random.shuffle(b)
    print(b)
    # result:[16 14 15 13 17 12 10 11 18 19]

"""