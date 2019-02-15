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

from util.coordinate_transform.transform import Transform, Vector, Rotation
#暂且用back的三个点形成的三角形中位线，这有待改进（例如改成有波动的），因为关系到模拟效果
def x_Rotation(pose, angel):
    pose_z = np.asarray(pose)
    #获取轴线
    axis = Vector(pose[18, :] - (pose[16, :]+pose[17, :])/2)
    R_matrix = Rotation(2*np.pi/360*angel, axis)
    #变成vector对象的数组
    pose_V = np.array([Vector(pose[i, :]) for i in range(20)])
    #旋转
    for i in range(20):
        pose_V[i] = R_matrix(pose_V[i])
        pose_z[i, :] = [pose_V[i].co[0], pose_V[i].co[1], pose_V[i].co[2]]
    #恢复成ndarray
    return pose_z

def y_Rotation():
    pass
#暂且用back的三个点的中点和中指指根作为轴线，这有待改进（例如改成有随机扰动的），因为关系到模拟效果
def z_Rotation(pose, angel):
    pose_z = np.asarray(pose)
    #获取轴线
    axis = Vector(pose[7, :] - pose[19, :])
    R_matrix = Rotation(2*np.pi/360*angel, axis)
    #变成vector对象的数组
    pose_V = np.array([Vector(pose[i, :]) for i in range(20)])
    #旋转
    for i in range(20):
        pose_V[i] = R_matrix(pose_V[i])
        pose_z[i, :] = [pose_V[i].co[0], pose_V[i].co[1], pose_V[i].co[2]]
    #恢复成ndarray
    return pose_z

file_name_all = file_name('/home/chen/Documents/.git/Mocap_SIG18_Data/training_data')
for file_name in file_name_all:
    onefile_data = read_file(file_name)
    file_name_detail = file_name.split('/')
    userid = file_name_detail[7]
    capid = file_name_detail[8]
    seqid = file_name_detail[9]
    pose_num = 0
    for pose_id in onefile_data:

        pose_id = np.reshape(pose_id, (19, 3))
        pose_id_back = (pose_id[16,:]+pose_id[17,:]+pose_id[18,:])/3.0
        pose_id_back = pose_id_back[np.newaxis, :]
        pose_id = np.concatenate([pose_id,pose_id_back])


        # 开始旋转
        for angel_j in range(360):
            x_Rotation(pose_id, 5)

            figure_joint_skeleton(pose_id, "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake/"
                                  +userid+"/"+capid+"/"+seqid+"/", pose_num+angel_j)
        pose_num = pose_num + 1
