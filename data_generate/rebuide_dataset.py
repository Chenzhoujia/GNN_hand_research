# -- coding: utf-8 --
import numpy as np
import os, random, struct, math
import matplotlib.pyplot as plt
from scipy import signal
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

b = np.load("t_a_sort.npy")
b_name = np.load("tremor_file_name_all.npy")

np.savetxt("t_a_sort.txt",b)
np.savetxt("tremor_file_name_all.txt",b_name)

np.save("tremor_file_name_all.npy",tremor_file_name_all)
pose_file_name_all = get_file_name('/home/chen/Documents/.git/Mocap_SIG18_Data/training_data',"trc")
tremor_file_name_all = get_file_name('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/extract_data',"txt")

tremor_file_name_all = np.array(tremor_file_name_all)
tremor_file_name_all = np.reshape(tremor_file_name_all,(-1,6))
draw_index = 0 # draw index
"""
#统计分析主动运动，与比例
v_b, v_a = signal.butter(8, 0.1, 'lowpass')#去除5HZ以上的
v_x = []
v_y = []
v_z = []

v_mx = []
v_my = []
v_mz = []

for pose_file_name_index, pose_file_name in enumerate(pose_file_name_all):
    pose_num = 0
    pose_file_name_detail = pose_file_name.split('/')
    pose_userid = pose_file_name_detail[7]
    pose_capid = pose_file_name_detail[8]
    pose_seqid = pose_file_name_detail[9]

    onefile_data = read_file(pose_file_name)
    onefile_data = onefile_data[:, 48:]

    if np.size(onefile_data, axis=0) <= 30:
        continue

    for f_i in range(9):
        onefile_data[:,f_i] = signal.filtfilt(v_b, v_a, onefile_data[:,f_i])
    onefile_data[:, 0] = (onefile_data[:, 0] +onefile_data[:, 3] +onefile_data[:, 6])/3
    onefile_data[:, 1] = (onefile_data[:, 1] +onefile_data[:, 4] +onefile_data[:, 7])/3
    onefile_data[:, 2] = (onefile_data[:, 2] +onefile_data[:, 5] +onefile_data[:, 8])/3
    # 统计最大最小值之间的差异。，统计运动距离
    onefile_data[:, 3] = onefile_data[:, 0]
    onefile_data[:, 4] = onefile_data[:, 1]
    onefile_data[:, 5] = onefile_data[:, 2]
    onefile_data[:, 3].sort()
    onefile_data[:, 4].sort()
    onefile_data[:, 5].sort()
    v_x.append(onefile_data[-1, 3] - onefile_data[0, 3])
    v_y.append(onefile_data[-1, 4] - onefile_data[0, 4])
    v_z.append(onefile_data[-1, 5] - onefile_data[0, 5])

    for f_i in range(np.size(onefile_data, axis=0)):
        if f_i ==0:
            onefile_data[f_i, 6] = 0.0
            onefile_data[f_i, 7] = 0.0
            onefile_data[f_i, 8] = 0.0
        else:
            onefile_data[f_i, 6] = abs(onefile_data[f_i, 0] - onefile_data[f_i - 1, 0])
            onefile_data[f_i, 7] = abs(onefile_data[f_i, 1] - onefile_data[f_i - 1, 1])
            onefile_data[f_i, 8] = abs(onefile_data[f_i, 2] - onefile_data[f_i - 1, 2])
    v_mx.append(np.max(onefile_data[:, 6]))
    v_my.append(np.max(onefile_data[:, 7]))
    v_mz.append(np.max(onefile_data[:, 8]))
    # 绘制3D图
    if pose_file_name_index%10==0:
        fig = plt.figure(1)
        fig.clear()
        ax1 = plt.subplot(111, projection='3d')
        ax1.plot(onefile_data[:, 0], onefile_data[:, 1], onefile_data[:, 2], linewidth=0.8)
        plt.savefig("/home/chen/Documents/GNN_hand_research/data_generate/image/" + str(pose_file_name_index).zfill(7) + ".png")
v_x = np.array(v_x)
v_x.sort()
v_y = np.array(v_y)
v_y.sort()
v_z = np.array(v_z)
v_z.sort()
v_mx = np.array(v_mx)
v_mx.sort()
v_my = np.array(v_my)
v_my.sort()
v_mz = np.array(v_mz)
v_mz.sort()

v_num = np.size(v_x, axis=0)
v_num = np.linspace(0,v_num,v_num)
fig = plt.figure(2)
fig.clear()
ax1 = plt.subplot(321)
ax1.plot(v_num,v_x)
ax1.set_title(v_x.mean())
ax1.grid(color='b',linestyle=':',linewidth=1)
ax2 = plt.subplot(323)
ax2.plot(v_num,v_y)
ax2.set_title(v_y.mean())
ax2.grid(color='b',linestyle=':',linewidth=1)
ax3 = plt.subplot(325)
ax3.plot(v_num,v_z)
ax3.set_title(v_z.mean())
ax3.grid(color='b',linestyle=':',linewidth=1)
ax4 = plt.subplot(322)
ax4.plot(v_num,v_mx)
ax4.set_title(v_mx.mean())
ax4.grid(color='b',linestyle=':',linewidth=1)
ax5 = plt.subplot(324)
ax5.plot(v_num,v_my)
ax5.set_title(v_my.mean())
ax5.grid(color='b',linestyle=':',linewidth=1)
ax6 = plt.subplot(326)
ax6.plot(v_num,v_mz)
ax6.set_title(v_mz.mean())
ax6.grid(color='b',linestyle=':',linewidth=1)

plt.show()
"""
t_a = []
t_m = []
for tremor_file_name_index, tremor_file_name in enumerate(tremor_file_name_all):
    cross_zero = 0
    # 验证，分割，读取，幅度变化
    #分割
    tremor_file_name_detail = tremor_file_name[0].split('/')
    tremor_capid = tremor_file_name_detail[8]
    # 手工筛选
    #if tremor_capid not in ["T034_Left","T033_Left"]:#"T002_Left","T040_Right","T055_Left","T020_Left","T008_Right","T001_Left"
    #    print("pass "+tremor_file_name[0] )
    #    continue

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
    #幅度统计
    tremor_xyz = np.sqrt(tremor_x**2 + tremor_y**2 + tremor_z**2)
    t_a.append(np.mean(tremor_xyz))
    t_m.append(np.max(tremor_xyz))

    """
    if tremor_file_name_index%10000==0:
        print(tremor_file_name_index)
        t_a = np.array(t_a)
        t_m = np.array(t_m)
        t_a_num = np.size(t_a, axis=0)
        t_a_num = np.linspace(0, t_a_num, t_a_num)

        fig = plt.figure(1)
        fig.clear()
        ax1 = plt.subplot(211)
        ax1.plot(t_a_num, t_a, linewidth=0.8)
        ax1.grid(color='b', linestyle=':', linewidth=0.5)
        ax2 = plt.subplot(212)
        ax2.plot(t_a_num, t_m, linewidth=0.8)
        ax2.grid(color='b', linestyle=':', linewidth=0.5)
        plt.savefig("/home/chen/Documents/GNN_hand_research/data_generate/image/" + str(tremor_file_name_index).zfill(7) + ".png")
        t_a = t_a.tolist()
        t_m = t_m.tolist()
    """
t_a_sort = np.argsort(t_a)
np.save("t_a_sort.npy",t_a_sort)
tremor_file_name_all = tremor_file_name_all[t_a_sort]
np.save("tremor_file_name_all.npy",tremor_file_name_all)
"""
    #频率统计
    num_sample = 0
    tremor_data_pre = tremor_x[0]
    t_f = []
    for tremor_data in tremor_x:
        if tremor_data*tremor_data_pre<0:
            t_f.append(num_sample)
            num_sample = 0
        else:
            num_sample += 1
        tremor_data_pre = tremor_data
    t_f = np.array(t_f)
    t_f_num = np.size(t_f, axis=0)
    t_f_num = np.linspace(0, t_f_num, t_f_num)

    tremor_y_num = np.size(tremor_y, axis=0)
    tremor_y_num = np.linspace(0, tremor_y_num, tremor_y_num)
    fig = plt.figure(1)
    fig.clear()
    ax1 = plt.subplot(411)
    ax1.plot(tremor_y_num, tremor_x, linewidth=0.8)
    ax1.grid(color='b', linestyle=':', linewidth=0.5)
    ax2 = plt.subplot(412)
    ax2.plot(tremor_y_num, tremor_y, linewidth=0.8)
    ax2.grid(color='b', linestyle=':', linewidth=0.5)
    ax3 = plt.subplot(413)
    ax3.plot(tremor_y_num, tremor_z, linewidth=0.8)
    ax3.grid(color='b', linestyle=':', linewidth=0.5)

    ax4 = plt.subplot(414)
    ax4.plot(t_f_num, t_f, linewidth=0.8)
    ax4.grid(color='b', linestyle=':', linewidth=0.5)
    ax4.set_title(np.mean(t_f))
    plt.savefig(
        "/home/chen/Documents/GNN_hand_research/data_generate/image/" + str(tremor_file_name_index).zfill(7) + ".png")

    #幅度变化

    modify_amp = 10000
    tremor_x /= modify_amp
    tremor_y /= modify_amp
    tremor_z /= modify_amp
    """

"""
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
            print("pose_time is: "+str(pose_time)+"tremor_time is:"+str(tremor_time))
            continue
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

        onefile_data_repeat = np.concatenate((onefile_data_repeat, onefile_data_repeat_saved), axis=1)

        # 绘制3D图
        fig = plt.figure(1)
        fig.clear()
        ax1 = plt.subplot(131, projection = '3d')
        ax2 = plt.subplot(132, projection = '3d')
        ax3 = plt.subplot(133, projection = '3d')
        ax1.plot(onefile_data_repeat_saved[:, 0],onefile_data_repeat_saved[:, 1],onefile_data_repeat_saved[:, 2],linewidth = 0.3)
        ax2.plot(onefile_xyz_axis_repeat[:, 0],onefile_xyz_axis_repeat[:, 1],onefile_xyz_axis_repeat[:, 2],linewidth = 0.3)
        ax3.plot(onefile_data_repeat[:, 0], onefile_data_repeat[:, 1], onefile_data_repeat[:, 2],linewidth = 0.3)
        plt.savefig("/home/chen/Documents/GNN_hand_research/data_generate/image/" + str(draw_index).zfill(7) + ".png")
        draw_index = draw_index+1
        # 分析统计幅度和频率

        #保存txt文件
        path_r_txt_shake = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra/"\
                     +tremor_capid+ "/" + tremor_seqid+"/"+pose_userid +"/"+pose_capid + "/"

        if not os.path.isdir(path_r_txt_shake):
            os.makedirs(path_r_txt_shake)
            print("creat path : " + path_r_txt_shake)

        np.savetxt(path_r_txt_shake + pose_seqid + ".txt", onefile_data_repeat)  # 缺省按照'%.18e'格式保存数据，以空格分隔
"""
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