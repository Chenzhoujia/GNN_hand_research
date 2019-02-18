# -- coding: utf-8 --
import numpy as np
import os, random, struct
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
    pose_x = np.asarray(pose)
    #获取轴线
    axis = Vector(pose_x[18, :] - (pose_x[16, :]+pose_x[17, :])/2)
    R_matrix = Rotation(2*np.pi/360*angel, axis)
    #变成vector对象的数组
    pose_V = np.array([Vector(pose_x[i, :]) for i in range(20)])
    #旋转
    for i in range(20):
        pose_V[i] = R_matrix(pose_V[i])
        pose_x[i, :] = [pose_V[i].co[0], pose_V[i].co[1], pose_V[i].co[2]]
    #恢复成ndarray
    return pose_x
# 暂且用back的三个点形成的中点（19）与小拇指指根（13）、食指指根（4）形成的平面的垂线，这有待改进（例如改成有波动的），因为关系到模拟效果
# (a0,a1,a2)x(b0,b1,b2)=(a1b2-a2b1,a2b0-a0b2,a0b1-a1b0)
def y_Rotation(pose, angel):
    pose_y = np.asarray(pose)
    #获取轴线
    ox = pose_y[13, :] - pose_y[19, :]
    oy = pose_y[4 , :] - pose_y[19, :]
    axis = Vector(np.array([ox[1] * oy[2] - ox[2] * oy[1],
                     ox[2] * oy[0] - ox[0] * oy[2],
                     ox[0] * oy[1] - ox[1] * oy[0]]))
    R_matrix = Rotation(2*np.pi/360*angel, axis)
    #变成vector对象的数组
    pose_V = np.array([Vector(pose_y[i, :]) for i in range(20)])
    #旋转
    for i in range(20):
        pose_V[i] = R_matrix(pose_V[i])
        pose_y[i, :] = [pose_V[i].co[0], pose_V[i].co[1], pose_V[i].co[2]]
    #恢复成ndarray
    return pose_y

#暂且用back的三个点的中点和中指指根作为轴线，这有待改进（例如改成有随机扰动的），因为关系到模拟效果
def z_Rotation(pose, angel):
    pose_z = np.asarray(pose)
    #获取轴线
    axis = Vector(pose_z[7, :] - pose_z[19, :])
    R_matrix = Rotation(2*np.pi/360*angel, axis)
    #变成vector对象的数组
    pose_V = np.array([Vector(pose_z[i, :]) for i in range(20)])
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
    step_num = 0

    # 根据视频设计震颤函数 按照120hz，4~6HZ, 决定振幅,频率 采样震颤函数 形成数组
    rot_ax = 2#0 -> x ; 1 -> y ; 2 -> z
    amp = random.uniform(30, 60)
    fre = random.uniform(4, 6)
    time_step = np.arange(0, np.pi * 2, np.pi * 2 / (120/fre))
    tremor_amplitude = amp * np.sin(time_step)


    onefile_data_pose = []
    for pose_id in onefile_data:
        pose_id = np.reshape(pose_id, (19, 3))
        pose_id_back = (pose_id[16,:]+pose_id[17,:]+pose_id[18,:])/3.0
        pose_id_back = pose_id_back[np.newaxis, :]
        pose_id = np.concatenate([pose_id,pose_id_back])
        #pose_saved = np.asarray(pose_id)

        #从震颤数组中采样选取幅度
        angel_cur = tremor_amplitude[pose_num]
        pose_num = pose_num + 1
        step_num = step_num + 1
        if pose_num==time_step.size:
            pose_num = 0

        # 开始旋转
        #figure_joint_skeleton(pose_id, "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake/"
        #                      +userid+"/"+capid+"/"+seqid+"/", step_num)
        if rot_ax==0:
            x_Rotation(pose_id, angel_cur)
        elif  rot_ax==1:
            y_Rotation(pose_id, angel_cur)
        else:
            z_Rotation(pose_id, angel_cur)
        #figure_joint_skeleton(pose_id, "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake/"
        #                      +userid+"/"+capid+"/"+seqid+"/", step_num+1000)

        rot_inf = np.array([np.float64(amp), np.float64(fre), angel_cur])
        rot_inf2 = np.array([np.float64(rot_ax), np.float64(0), np.float64(0)])
        pose_id = np.concatenate([pose_id, rot_inf[np.newaxis, :], rot_inf2[np.newaxis, :]])
        onefile_data_pose.append(pose_id)
    onefile_data_pose = np.array(onefile_data_pose)
    #保存txt文件
    path = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake/"+userid+"/"+capid
    if not os.path.isdir(path):
        os.makedirs(path)
        print("creat path : " + path)
    shape = onefile_data_pose.shape
    #np.savetxt(path + "/" + seqid+".txt", onefile_data_pose.reshape(shape[0], -1))  # 缺省按照'%.18e'格式保存数据，以空格分隔
    # read test

    onefile_data_pose_r = np.loadtxt(path +"/" + seqid+".txt")
    onefile_data_pose_r = onefile_data_pose_r.reshape(shape[0],shape[1],shape[2])
    onefile_data_pose_r = onefile_data_pose_r[:,0:shape[1]-2,:]
    for test_read_i in range(shape[0]):
        figure_joint_skeleton(onefile_data_pose_r[test_read_i,:,:] ,path + "/"+seqid+"/", test_read_i)
    