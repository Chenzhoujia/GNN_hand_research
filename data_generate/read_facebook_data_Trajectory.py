# -- coding: utf-8 --
import numpy as np
import os, random, struct, math
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

file_name_all = get_file_name('/home/chen/Documents/.git/Mocap_SIG18_Data/training_data',"trc")
tremor_file_name_all = get_file_name('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/extract_data',"txt")
i_userid_sel_list = ["T008"]
i_seqid_sel_list = ["2_Hz_lower","Counting","Finger_tapping","Hands_in_pronation","Months_backward","Playing_piano","Thumbs_up","Top_nose_left","Top_nose_right","Top_top"]
for i_userid_sel in i_userid_sel_list:
    for i_seqid_sel in i_seqid_sel_list:
        for file_name in file_name_all:
            onefile_data = read_file(file_name)
            file_name_detail = file_name.split('/')
            userid = file_name_detail[7]
            capid = file_name_detail[8]
            seqid = file_name_detail[9]
            pose_num = 0

            # 根据视频设计震颤函数 按照120hz，4~6HZ, 决定振幅,频率 采样震颤函数 形成数组
            """
            rot_ax = 0 #0 -> x ; 1 -> y ; 2 -> z
            amp = random.uniform(30, 60)
            fre = random.uniform(4, 6)
            time_step = np.arange(0, np.pi * 2, np.pi * 2 / (120/fre))
            tremor_amplitude = amp * np.sin(time_step)
            """
            #获取已有的完整xyz三轴的震荡曲线


            tremor_file_name_all_tsv = []
            for i in tremor_file_name_all:
                i_detail = i.split('/')
                i_userid = i_detail[8]
                i_seqid = i_detail[9]
                if i.endswith("txt") and i_userid.startswith(i_userid_sel) and i_seqid.startswith(i_seqid_sel):
                    tremor_file_name_all_tsv.append(i)
            tremor_file_name_all = tremor_file_name_all_tsv
            tremor_file_name_all_tsv = []

            if len(tremor_file_name_all)!=6:
                raise RuntimeError("file name is"+tremor_file_name_all)
            else:
                #: (0) sensor-1 x, (1) sensor-1 y,(2) sensor-1 z, (4) sensor-2 x, (5) sensor-2 y, (6) sensor-2 z
                # 手背是sensor-1，手臂是sensor-2
                if tremor_file_name_all[0].endswith("00.txt") and tremor_file_name_all[1].endswith("01.txt") and tremor_file_name_all[2].endswith("02.txt"):
                    tremor_x = np.loadtxt(tremor_file_name_all[0])
                    tremor_y = np.loadtxt(tremor_file_name_all[1])
                    tremor_z = np.loadtxt(tremor_file_name_all[2])
                else:
                    raise RuntimeError("wrong order")

                if np.size(tremor_x)!=np.size(tremor_y)!=np.size(tremor_z):
                    raise RuntimeError("XYZ different size")


            #时间匹配
            tremor_time = np.size(tremor_x)/1000.0
            pose_time = np.size(onefile_data,0)/120.0
            if pose_time>tremor_time:
                raise RuntimeError("pose_time is: "+str(pose_time)+"tremor_time is:"+str(tremor_time))
            repeat_time = tremor_time/pose_time
            repeat_time = int(math.floor(repeat_time))
            #onefile_data_repeat = np.zeros((repeat_time*np.size(onefile_data,0), np.size(onefile_data,1)))
            onefile_data_reverse = np.zeros((np.size(onefile_data, 0), np.size(onefile_data, 1)))
            for trverse_i in range(np.size(onefile_data, 0)):
                onefile_data_reverse[trverse_i,:] = onefile_data[np.size(onefile_data, 0) - trverse_i -1,:]
            onefile_data_repeat = onefile_data
            for repeat_time_i in range(repeat_time-1):
                if repeat_time_i%2==0:
                    onefile_data_repeat = np.concatenate((onefile_data_repeat, onefile_data_reverse), axis=0)
                else:
                    onefile_data_repeat = np.concatenate((onefile_data_repeat, onefile_data), axis=0)
            #采样tremor
            sampl_interval = int(math.floor(1000.0/120.0))
            if np.size(onefile_data_repeat, 0)*sampl_interval>np.size(tremor_x):
                raise RuntimeError("too much sample point, which is wired")
            onefile_data_repeat_ampx = np.zeros(np.size(onefile_data_repeat, 0))
            onefile_data_repeat_ampy = np.zeros(np.size(onefile_data_repeat, 0))
            onefile_data_repeat_ampz = np.zeros(np.size(onefile_data_repeat, 0))
            for amp_i in range(np.size(onefile_data_repeat, 0)):
                onefile_data_repeat_ampx[amp_i] = tremor_x[amp_i * sampl_interval]
                onefile_data_repeat_ampy[amp_i] = tremor_y[amp_i * sampl_interval]
                onefile_data_repeat_ampz[amp_i] = tremor_z[amp_i * sampl_interval]
            #幅度变换
            onefile_data_repeat_ampx = onefile_data_repeat_ampx / 2500000.0 * 45.0  #左右 -> y_Rotation
            onefile_data_repeat_ampy = onefile_data_repeat_ampy / 2500000.0 * 45.0  #y-axis pointing towards the fingers.
            onefile_data_repeat_ampz = onefile_data_repeat_ampz / 2500000.0 * 45.0  #z-axis pointing away from the skin surface 上下 -> x_Rotation
            onefile_data_pose = []
            for pose_id in onefile_data_repeat:
                pose_id = np.reshape(pose_id, (19, 3))
                pose_id_back = (pose_id[16,:]+pose_id[17,:]+pose_id[18,:])/3.0
                pose_id_back = pose_id_back[np.newaxis, :]
                pose_id = np.concatenate([pose_id,pose_id_back])
                #pose_saved = np.asarray(pose_id)

                #从震颤数组中采样选取幅度
                angel_cur_x = onefile_data_repeat_ampx[pose_num]
                angel_cur_y = onefile_data_repeat_ampy[pose_num]
                angel_cur_z = onefile_data_repeat_ampz[pose_num]
                pose_num = pose_num + 1
                # 开始旋转
                x_Rotation(pose_id, angel_cur_z)
                y_Rotation(pose_id, angel_cur_x)
                #z_Rotation(pose_id, angel_cur_z)

                rot_inf = np.array([np.float64(angel_cur_z), np.float64(angel_cur_x), np.float64(0)])

                pose_id = np.concatenate([pose_id, rot_inf[np.newaxis, :]])
                onefile_data_pose.append(pose_id)
            onefile_data_pose = np.array(onefile_data_pose)
            #保存txt文件
            path_r_txt = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/shake/"+i_userid_sel+ "/" + i_seqid_sel+"/"+userid +"/"+capid
            if not os.path.isdir(path_r_txt):
                os.makedirs(path_r_txt)
                print("creat path : " + path_r_txt)
            shape = onefile_data_pose.shape
            np.savetxt(path_r_txt + "/" + seqid + ".txt", onefile_data_pose.reshape(shape[0], -1))  # 缺省按照'%.18e'格式保存数据，以空格分隔
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