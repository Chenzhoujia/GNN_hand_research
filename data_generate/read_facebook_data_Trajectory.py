# -- coding: utf-8 --
import numpy as np
import os, random, struct, math
from util.visual import figure_joint_skeleton, figure_hand_back
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


def get_file_name(file_dir,suffix=None):
    file_names = []
    for root, dirs, files in os.walk(file_dir):
        if files==[]:
            continue
        for file in files:
            if suffix==None:
                file_names.append(root+"/"+file)  # 当前目录路径
            elif file.endswith("."+suffix):
                file_names.append(root + "/" + file)  # 当前目录路径

    return file_names
        #print('sub_dirs:', dirs)  # 当前路径下所有子目录
        #print('files:', files)  # 当前路径下所有非目录子文件

from util.coordinate_transform.transform import Transform, Vector, Rotation
#暂且用back的三个点形成的三角形中位线，这有待改进（例如改成有波动的），因为关系到模拟效果
# 上下扑动
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
#   左右晃动
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
# 翻滚
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

pose_file_name_all = get_file_name('/home/chen/Documents/.git/Mocap_SIG18_Data/training_data',"trc")
tremor_file_name_all = get_file_name('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/extract_data',"txt")

tremor_file_name_all = np.array(tremor_file_name_all)
tremor_file_name_all = np.reshape(tremor_file_name_all,(-1,6))
for tremor_file_name_index, tremor_file_name in enumerate(tremor_file_name_all):
    # 验证，分割，读取，幅度变化
    #分割
    tremor_file_name_detail = tremor_file_name[0].split('/')
    tremor_capid = tremor_file_name_detail[8]
    tremor_seqid = tremor_file_name_detail[9]
    #验证
    exame_id = 0
    for i in [0,1,2,4,5,6]:
        exam_detail = tremor_file_name[exame_id].split('/')
        exam_capid = exam_detail[8]
        exam_seqid = exam_detail[9]
        exam_xyzid = exam_detail[10]
        if exam_capid!=tremor_capid or exam_seqid!=tremor_seqid:
            raise RuntimeError("验证文件名时出错")
        if exam_xyzid!=str(i).zfill(2)+".txt":
            raise RuntimeError("验证文件名时出错")
        exame_id+=1
    #读取
    tremor_x = np.loadtxt(tremor_file_name[0])
    tremor_y = np.loadtxt(tremor_file_name[1]) #y-axis pointing towards the fingers.
    tremor_z = np.loadtxt(tremor_file_name[2]) #z-axis pointing away from the skin surface
    #幅度变化
    modify_amp = 5000
    tremor_x /= modify_amp
    tremor_y /= modify_amp
    tremor_z /= modify_amp

    for pose_file_name in pose_file_name_all:

        pose_num = 0
        pose_file_name_detail = pose_file_name.split('/')
        pose_userid = pose_file_name_detail[7]
        pose_capid = pose_file_name_detail[8]
        pose_seqid = pose_file_name_detail[9]

        onefile_data = read_file(pose_file_name)
        onefile_data = onefile_data[:,48:]

        # 为了匹配如下加速度计的方向的单位向量在pose坐标系中的分量
        # y-axis pointing towards the fingers.
        # z-axis pointing away from the skin surface
        for onefile_data_i_index, onefile_data_i in enumerate(onefile_data):
            # 暂且用back的三个点形成的三角形中位线作为x-axis
            onefile_data_i = np.reshape(onefile_data_i, (3, 3))
            x_axis = np.array(onefile_data_i[2, :] - (onefile_data_i[1, :] + onefile_data_i[0, :]) / 2)
            x_axis = x_axis/np.linalg.norm(x_axis, ord=2)

            y_axis = np.array(onefile_data_i[1, :] - onefile_data_i[0, :])
            y_axis = y_axis / np.linalg.norm(y_axis, ord=2)

            z_axis_ox = onefile_data_i[2, :] - onefile_data_i[1, :]
            z_axis_oy = onefile_data_i[0, :] - onefile_data_i[1, :]
            z_axis = np.array([z_axis_ox[1] * z_axis_oy[2] - z_axis_ox[2] * z_axis_oy[1],
                              z_axis_ox[2] * z_axis_oy[0] - z_axis_ox[0] * z_axis_oy[2],
                              z_axis_ox[0] * z_axis_oy[1] - z_axis_ox[1] * z_axis_oy[0]])
            z_axis = z_axis / np.linalg.norm(z_axis, ord=2)
            xyz_axis = np.concatenate((x_axis, y_axis), axis=0)
            xyz_axis = np.concatenate((xyz_axis, z_axis), axis=0)

            if onefile_data_i_index==0:
                onefile_xyz_axis = np.expand_dims(xyz_axis, axis=0)
            else:
                onefile_xyz_axis = np.concatenate((onefile_xyz_axis, np.expand_dims(xyz_axis, axis=0)), axis=0)

        #onefile_data = np.reshape(onefile_data, (-1 ,3, 3))

        #时间匹配
        tremor_time = np.size(tremor_x)/1000.0
        pose_time = np.size(onefile_data,0)/120.0
        if pose_time>tremor_time:
            raise RuntimeError("pose_time is: "+str(pose_time)+"tremor_time is:"+str(tremor_time))
        repeat_time = tremor_time/pose_time
        repeat_time = int(math.floor(repeat_time))

            #pose
        onefile_data_reverse = np.zeros((np.size(onefile_data, 0), np.size(onefile_data, 1)))
        for trverse_i in range(np.size(onefile_data, 0)):
            onefile_data_reverse[trverse_i,:] = onefile_data[np.size(onefile_data, 0) - trverse_i -1,:]

        onefile_data_repeat = onefile_data
        for repeat_time_i in range(repeat_time-1):
            if repeat_time_i%2==0:
                onefile_data_repeat = np.concatenate((onefile_data_repeat, onefile_data_reverse), axis=0)
            else:
                onefile_data_repeat = np.concatenate((onefile_data_repeat, onefile_data), axis=0)
            #坐标轴
        onefile_xyz_axis_reverse = np.zeros((np.size(onefile_xyz_axis, 0), np.size(onefile_xyz_axis, 1)))
        for trverse_i in range(np.size(onefile_xyz_axis, 0)):
            onefile_xyz_axis_reverse[trverse_i, :] = onefile_xyz_axis[np.size(onefile_xyz_axis, 0) - trverse_i - 1, :]

        onefile_xyz_axis_repeat = onefile_xyz_axis
        for repeat_time_i in range(repeat_time - 1):
            if repeat_time_i % 2 == 0:
                onefile_xyz_axis_repeat = np.concatenate((onefile_xyz_axis_repeat, onefile_xyz_axis_reverse), axis=0)
            else:
                onefile_xyz_axis_repeat = np.concatenate((onefile_xyz_axis_repeat, onefile_xyz_axis), axis=0)


        #采样tremor
        sampl_interval = 8#int(math.floor(1000.0/120.0))
        if np.size(onefile_data_repeat, 0)*sampl_interval>np.size(tremor_x):
            raise RuntimeError("too much sample point, which is wired")
        onefile_data_repeat_ampx = np.zeros(np.size(onefile_data_repeat, 0))
        onefile_data_repeat_ampy = np.zeros(np.size(onefile_data_repeat, 0))
        onefile_data_repeat_ampz = np.zeros(np.size(onefile_data_repeat, 0))
        for amp_i in range(np.size(onefile_data_repeat, 0)):
            onefile_data_repeat_ampx[amp_i] = tremor_x[amp_i * sampl_interval]
            onefile_data_repeat_ampy[amp_i] = tremor_y[amp_i * sampl_interval]
            onefile_data_repeat_ampz[amp_i] = tremor_z[amp_i * sampl_interval]

        #将加速度计的坐标轴的三个振幅x'y'z'，映射到世界坐标系上(xyz)(xyz)(xyz)
        for mapping_i in range(3):
            onefile_xyz_axis_repeat[:, mapping_i] *= onefile_data_repeat_ampx
            onefile_xyz_axis_repeat[:, mapping_i+3] *= onefile_data_repeat_ampy
            onefile_xyz_axis_repeat[:, mapping_i+6] *= onefile_data_repeat_ampz

        onefile_xyz_axis_repeat[:, 0] = onefile_xyz_axis_repeat[:, 0] + onefile_xyz_axis_repeat[:,
                                                                        3] + onefile_xyz_axis_repeat[:, 6]
        onefile_xyz_axis_repeat[:, 1] = onefile_xyz_axis_repeat[:, 1] + onefile_xyz_axis_repeat[:,
                                                                        4] + onefile_xyz_axis_repeat[:, 7]
        onefile_xyz_axis_repeat[:, 2] = onefile_xyz_axis_repeat[:, 2] + onefile_xyz_axis_repeat[:,
                                                                        5] + onefile_xyz_axis_repeat[:, 8]

        #交配 onefile_data_repeat(x,9)  tremor_x(x),根据可视化效果看看是否需要调整坐标系匹配
        onefile_data_repeat_saved = np.array(onefile_data_repeat)

        onefile_data_repeat[:, 0] += onefile_xyz_axis_repeat[:, 0]
        onefile_data_repeat[:, 3] += onefile_xyz_axis_repeat[:, 0]
        onefile_data_repeat[:, 6] += onefile_xyz_axis_repeat[:, 0]

        onefile_data_repeat[:, 1] += onefile_xyz_axis_repeat[:, 1]
        onefile_data_repeat[:, 4] += onefile_xyz_axis_repeat[:, 1]
        onefile_data_repeat[:, 7] += onefile_xyz_axis_repeat[:, 1]

        onefile_data_repeat[:, 2] += onefile_xyz_axis_repeat[:, 2]
        onefile_data_repeat[:, 5] += onefile_xyz_axis_repeat[:, 2]
        onefile_data_repeat[:, 8] += onefile_xyz_axis_repeat[:, 2]

        #保存txt文件
        path_r_txt = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra/"\
                     +tremor_capid+ "/" + tremor_seqid+"/"+pose_userid +"/"+pose_capid
        if not os.path.isdir(path_r_txt):
            os.makedirs(path_r_txt)
            print("creat path : " + path_r_txt)
        np.savetxt(path_r_txt + "/" + pose_seqid + "_shake.txt", onefile_data_repeat)  # 缺省按照'%.18e'格式保存数据，以空格分隔

        np.savetxt(path_r_txt + "/" + pose_seqid + "_gt.txt", onefile_data_repeat_saved)  # 缺省按照'%.18e'格式保存数据，以空格分隔

        """
        #通过轨迹效果调整坐标轴和缩放尺度
        path_view_txt = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra/view/"\
                     +tremor_capid+ "/" + tremor_seqid+"/"+pose_userid +"/"+pose_capid + "/" + pose_seqid+ "/"
        if not os.path.isdir(path_view_txt):
            os.makedirs(path_view_txt)
            print("creat path : " + path_view_txt)
        shape_onefile_data_repeat = np.size(onefile_data_repeat,0)
        for test_read_i in range(shape_onefile_data_repeat):
            figure_hand_back(onefile_data_repeat[test_read_i,:],onefile_data_repeat_saved[test_read_i,:] ,path_view_txt, test_read_i)
        """
        """
        # read test
        #onefile_data_pose_r = np.loadtxt(path +"/" + seqid+".txt")
        onefile_data_pose_r = onefile_data_pose
        onefile_data_pose_r = onefile_data_pose_r.reshape(shape[0],shape[1],shape[2])
        onefile_data_pose_r = onefile_data_pose_r[:,0:shape[1]-1,:]
        path = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake_view/"+userid+"/"+capid
        if not os.path.isdir(path):
            os.makedirs(path)
            print("creat path : " + path)
        for test_read_i in range(shape[0]):
            figure_joint_skeleton(onefile_data_pose_r[test_read_i,:,:] ,path + "/"+seqid+"/", test_read_i)
        """