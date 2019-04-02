#coding=utf-8
import pickle, os, random, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal,interpolate
from time import sleep
from tqdm import tqdm

#主动运动的类，功能包括：对象保存、文件读取、信号滤波、绘图可视化、统计分析
class volunter_motion(object):
    address = ""       # 存放所有数据的root地址
    save_file = ""     # 为了节省时间，保存对象的地址
    pose_name = []
    image_path = ""
    pose_all = []      # 存放所有的pose（过滤之后的）
    average_speed = []
    to_cm_rate = 0
    allfile_xyz_axis = []
    def __init__(self,address_,save_file_,image_path_,to_cm_rate_):
        self.address = address_
        self.save_file = save_file_
        self.pose_name = self.__get_file_name("trc")
        self.image_path = image_path_
        self.to_cm_rate = to_cm_rate_

        # 清道夫的工作
        if not os.path.exists(self.image_path + "/volunter_move/"):
            os.makedirs(self.image_path + "/volunter_move/")
            print('create ' + self.image_path + "/volunter_move/")
        volunter_motion.del_file(self.image_path + "/volunter_move/")

        self.average_speed = np.load("./Intermediate_results/volunter_average_speed.npy")
        self.average_speed = self.average_speed/self.to_cm_rate # 主动运动算的时候没有除
        self.read_all_pose()

        #self.compute_axis()

    # 保存当前对象
    def save_this(self):
        pickle_file = open(self.save_file, 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()
    # 加载这个类的一个对象，节省时间
    @ staticmethod
    def lode_this(save_file):
        pockle_file = open(save_file, 'rb')
        return pickle.load(pockle_file)
    @ staticmethod
    def del_file(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                volunter_motion.del_file(c_path)
            else:
                os.remove(c_path)
    #读取文件名问filename的一个pose
    def read_one_pose_file(self, filename):
        pos = []
        with open(filename, 'r') as file_to_read:
            for i in range(5):
                lines = file_to_read.readline()
            while True:
                lines = file_to_read.readline()
                lines = lines.split('\t')
                lines = lines[2:-1]
                if len(lines) != 57:
                    break
                    pass
                p_tmp = [float(i) for i in lines]
                pos.append(p_tmp)
                pass
            pos = np.array(pos)
            return pos
    # 获取address目录下的所有以suffix结尾的文件
    def __get_file_name(self, suffix=None):
        file_names = []
        for root, dirs, files in os.walk(self.address):
            if files == []:
                continue
            for file in files:
                if suffix == None:
                    file_names.append(root + "/" + file)  # 当前目录路径
                elif file.endswith("." + suffix):
                    file_names.append(root + "/" + file)  # 当前目录路径

        return file_names
    @ staticmethod
    def three_distance(x1,y1,z1,x2,y2,z2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    # 读取所有pose、过滤掉高频成分、并可视化、保存
    def read_all_pose(self):
        v_b, v_a = signal.butter(8, 0.1, 'lowpass')  # 去除5HZ以上的

        for pose_file_name_index, pose_file_name in tqdm(enumerate(self.pose_name)):
            #检测时用，免得跑太久
            #if not pose_file_name_index%10 ==0:
            #    continue
            pose_num = 0
            pose_file_name_detail = pose_file_name.split('/')
            pose_userid = pose_file_name_detail[7]
            pose_capid = pose_file_name_detail[8]
            pose_seqid = pose_file_name_detail[9]

            onefile_data = self.read_one_pose_file(pose_file_name)
            onefile_data = onefile_data[:, 48:]

            # 太短以至于不能滤波
            if np.size(onefile_data, axis=0) <= 30:
                continue
            # 过滤
            for f_i in range(9):
                onefile_data[:, f_i] = signal.filtfilt(v_b, v_a, onefile_data[:, f_i])
            self.pose_all.append(onefile_data/self.to_cm_rate)

            #计算平均速度,算一次就行
            """
            speed = []
            v_mean_x = (onefile_data[:, 0] + onefile_data[:, 3] + onefile_data[:, 6]) / 3
            v_mean_y = (onefile_data[:, 1] + onefile_data[:, 4] + onefile_data[:, 7]) / 3
            v_mean_z = (onefile_data[:, 2] + onefile_data[:, 5] + onefile_data[:, 8]) / 3
            for one_position in range(np.size(onefile_data[:, 0])-1):
                speed.append(volunter_motion.three_distance(v_mean_x[one_position],v_mean_y[one_position],
                                                            v_mean_z[one_position],
                                                            v_mean_x[one_position + 1], v_mean_y[one_position + 1],
                                                            v_mean_z[one_position + 1]))
            self.average_speed.append(np.mean(np.array(speed)))
            """

            # 绘制3D图
            if pose_file_name_index % 10 == 0:
                fig = plt.figure(1)
                fig.clear()
                ax1 = plt.subplot(111, projection='3d')
                ax1.plot(onefile_data[:, 0], onefile_data[:, 1], onefile_data[:, 2], linewidth=0.8)
                plt.savefig(self.image_path + "/volunter_move/" + str(pose_file_name_index).zfill(7) + ".png")
        #self.average_speed = np.array(self.average_speed)
        #np.save("./Intermediate_results/volunter_average_speed.npy", self.average_speed)
    # 统计分析主动运动，与比例
    def __static(self):
        pass
        """
        # 求中心点
        onefile_data[:, 0] = (onefile_data[:, 0] + onefile_data[:, 3] + onefile_data[:, 6]) / 3
        onefile_data[:, 1] = (onefile_data[:, 1] + onefile_data[:, 4] + onefile_data[:, 7]) / 3
        onefile_data[:, 2] = (onefile_data[:, 2] + onefile_data[:, 5] + onefile_data[:, 8]) / 3
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
            if f_i == 0:
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
        if pose_file_name_index % 10 == 0:
            fig = plt.figure(1)
            fig.clear()
            ax1 = plt.subplot(111, projection='3d')
            ax1.plot(onefile_data[:, 0], onefile_data[:, 1], onefile_data[:, 2], linewidth=0.8)
            plt.savefig("/home/chen/Documents/GNN_hand_research/data_generate/image/" + str(pose_file_name_index).zfill(
                7) + ".png")
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
    v_num = np.linspace(0, v_num, v_num)
    fig = plt.figure(2)
    fig.clear()
    ax1 = plt.subplot(321)
    ax1.plot(v_num, v_x)
    ax1.set_title(v_x.mean())
    ax1.grid(color='b', linestyle=':', linewidth=1)
    ax2 = plt.subplot(323)
    ax2.plot(v_num, v_y)
    ax2.set_title(v_y.mean())
    ax2.grid(color='b', linestyle=':', linewidth=1)
    ax3 = plt.subplot(325)
    ax3.plot(v_num, v_z)
    ax3.set_title(v_z.mean())
    ax3.grid(color='b', linestyle=':', linewidth=1)
    ax4 = plt.subplot(322)
    ax4.plot(v_num, v_mx)
    ax4.set_title(v_mx.mean())
    ax4.grid(color='b', linestyle=':', linewidth=1)
    ax5 = plt.subplot(324)
    ax5.plot(v_num, v_my)
    ax5.set_title(v_my.mean())
    ax5.grid(color='b', linestyle=':', linewidth=1)
    ax6 = plt.subplot(326)
    ax6.plot(v_num, v_mz)
    ax6.set_title(v_mz.mean())
    ax6.grid(color='b', linestyle=':', linewidth=1)

    plt.show()
    """

    def compute_axis(self):
        # 为了匹配如下加速度计的方向的单位向量在pose坐标系中的分量
        # y-axis pointing towards the fingers.
        # z-axis pointing away from the skin surface
        for one_pose in tqdm(self.pose_all):
            for onefile_data_i_index, onefile_data_i in enumerate(one_pose):
                # 暂且用back的三个点形成的三角形中位线作为x-axis
                onefile_data_i = np.reshape(onefile_data_i, (3, 3))
                x_axis = np.array(onefile_data_i[2, :] - (onefile_data_i[1, :] + onefile_data_i[0, :]) / 2)
                x_axis = x_axis / np.linalg.norm(x_axis, ord=2)

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

                if onefile_data_i_index == 0:
                    onefile_xyz_axis = np.expand_dims(xyz_axis, axis=0)
                else:
                    onefile_xyz_axis = np.concatenate((onefile_xyz_axis, np.expand_dims(xyz_axis, axis=0)), axis=0)
            self.allfile_xyz_axis.append(onefile_xyz_axis)
    @ staticmethod
    def compute_one_axis(one_pose):
        # 为了匹配如下加速度计的方向的单位向量在pose坐标系中的分量
        # y-axis pointing towards the fingers.
        # z-axis pointing away from the skin surface
        for onefile_data_i_index, onefile_data_i in enumerate(one_pose):
            # 暂且用back的三个点形成的三角形中位线作为x-axis
            onefile_data_i = np.reshape(onefile_data_i, (3, 3))
            x_axis = np.array(onefile_data_i[2, :] - (onefile_data_i[1, :] + onefile_data_i[0, :]) / 2)
            x_axis = x_axis / np.linalg.norm(x_axis, ord=2)

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

            if onefile_data_i_index == 0:
                onefile_xyz_axis = np.expand_dims(xyz_axis, axis=0)
            else:
                onefile_xyz_axis = np.concatenate((onefile_xyz_axis, np.expand_dims(xyz_axis, axis=0)), axis=0)
        return onefile_xyz_axis

#主动运动的类，功能包括：对象保存、文件读取、信号滤波、绘图可视化、统计分析
class tremor_motion(object):
    address = ""        # 存放所有数据的root地址
    save_file = ""      # 为了节省时间，保存对象的地址
    pose_name = []
    sort_pose_name = []
    image_path = ""
    tremor_x = []       # 存放所有的pose
    tremor_y = []  # 存放所有的pose
    tremor_z = []  # 存放所有的pose
    tremor_amp = []
    tremor_amp_avg = []
    average_speed = []
    to_cm_rate = 0
    def __init__(self,address_,save_file_,image_path_, to_cm_rate_):
        self.address = address_
        self.save_file = save_file_
        self.pose_name = self.__get_file_name("txt")
        self.pose_name = np.array(self.pose_name)
        self.pose_name = np.reshape(self.pose_name, (-1, 6))
        self.image_path = image_path_
        self.sort_pose_name = np.load("./Intermediate_results/tremor_file_name_all.npy")
        self.average_speed = np.load("./Intermediate_results/tremor_average_speed.npy")
        self.to_cm_rate = to_cm_rate_

        # 清道夫的工作
        if not os.path.exists(self.image_path + "/tremor_move/"):
            os.makedirs(self.image_path + "/tremor_move/")
            print('create ' + self.image_path + "/tremor_move/")
        volunter_motion.del_file(self.image_path + "/tremor_move/")

        self.read_all_pose()
    # 保存当前对象
    def save_this(self):
        pickle_file = open(self.save_file, 'wb')
        pickle.dump(self, pickle_file)
        pickle_file.close()
    # 加载这个类的一个对象，节省时间
    @ staticmethod
    def lode_this(save_file):
        pockle_file = open(save_file, 'rb')
        return pickle.load(pockle_file)
    @ staticmethod
    def del_file(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                volunter_motion.del_file(c_path)
            else:
                os.remove(c_path)
    #读取文件名问filename的一个pose
    def read_one_pose_file(self, filename):
        pos = []
        with open(filename, 'r') as file_to_read:
            for i in range(5):
                lines = file_to_read.readline()
            while True:
                lines = file_to_read.readline()
                lines = lines.split('\t')
                lines = lines[2:-1]
                if len(lines) != 57:
                    break
                    pass
                p_tmp = [float(i) for i in lines]
                pos.append(p_tmp)
                pass
            pos = np.array(pos)
            return pos
    # 获取address目录下的所有以suffix结尾的文件
    def __get_file_name(self, suffix=None):
        file_names = []
        for root, dirs, files in os.walk(self.address):
            if files == []:
                continue
            for file in files:
                if suffix == None:
                    file_names.append(root + "/" + file)  # 当前目录路径
                elif file.endswith("." + suffix):
                    file_names.append(root + "/" + file)  # 当前目录路径

        return file_names


    # 读取所有pose、过滤掉高频成分、并可视化、保存
    def read_all_pose(self):
        for tremor_file_name_index, tremor_file_name in tqdm(enumerate(self.sort_pose_name)):
            #检测时用，免得跑太久
            #if tremor_file_name_index%10 != 0 and tremor_file_name_index<900:
            #    continue

            # 验证，分割，读取，幅度变化
            # 分割
            tremor_file_name_detail = tremor_file_name[0].split('/')
            tremor_capid = tremor_file_name_detail[8]
            # 手工筛选
            # if tremor_capid not in ["T034_Left","T033_Left"]:#"T002_Left","T040_Right","T055_Left","T020_Left","T008_Right","T001_Left"
            #    print("pass "+tremor_file_name[0] )
            #    continue
            tremor_seqid = tremor_file_name_detail[9]
            # 验证
            exame_id = 0
            for i in [0, 1, 2, 4, 5, 6]:
                exam_detail = tremor_file_name[exame_id].split('/')
                exam_capid = exam_detail[8]
                exam_seqid = exam_detail[9]
                exam_xyzid = exam_detail[10]
                if exam_capid != tremor_capid or exam_seqid != tremor_seqid:
                    raise RuntimeError("验证文件名时出错")
                if exam_xyzid != str(i).zfill(2) + ".txt":
                    raise RuntimeError("验证文件名时出错")
                exame_id += 1
            # 读取
            self.tremor_x.append(np.loadtxt(tremor_file_name[0])/self.to_cm_rate)
            self.tremor_y.append(np.loadtxt(tremor_file_name[1])/self.to_cm_rate)  # y-axis pointing towards the fingers.
            self.tremor_z.append(np.loadtxt(tremor_file_name[2])/self.to_cm_rate)  # z-axis pointing away from the skin surface
            #振幅
            self.tremor_amp.append(np.sqrt(self.tremor_x[-1]**2 + self.tremor_y[-1]**2 + self.tremor_z[-1]**2))
            self.tremor_amp_avg.append(np.mean(self.tremor_amp[-1]))

            #计算平均速度
            """
            speed = []
            for one_position in range(np.size(self.tremor_y[-1], axis=0)-1):
                speed.append(volunter_motion.three_distance(self.tremor_x[-1][one_position],self.tremor_y[-1][one_position],
                                                            self.tremor_z[-1][one_position],
                                                            self.tremor_x[-1][one_position + 1], self.tremor_y[-1][one_position + 1],
                                                            self.tremor_z[-1][one_position + 1]))
            self.average_speed.append(np.mean(np.array(speed)))
            """
            # 绘制3D图
            #if tremor_file_name_index % 10 == 0:
            if 1:
                tremor_y_num = np.size(self.tremor_y[-1], axis=0)
                tremor_y_num = np.linspace(0, tremor_y_num, tremor_y_num)
                fig = plt.figure(1)
                fig.clear()
                ax1 = plt.subplot(421)
                ax1.plot(tremor_y_num, self.tremor_x[-1], linewidth=0.8)
                ax1.grid(color='b', linestyle=':', linewidth=0.5)
                ax2 = plt.subplot(422)
                ax2.plot(tremor_y_num, self.tremor_y[-1], linewidth=0.8)
                ax2.grid(color='b', linestyle=':', linewidth=0.5)
                ax3 = plt.subplot(423)
                ax3.plot(tremor_y_num, self.tremor_z[-1], linewidth=0.8)
                ax3.grid(color='b', linestyle=':', linewidth=0.5)
                ax4 = plt.subplot(424)
                ax4.plot(tremor_y_num, self.tremor_amp[-1], linewidth=0.8)
                ax4.grid(color='b', linestyle=':', linewidth=0.5)
                ax4.set_title(self.tremor_amp_avg[-1])
                # 绘制3D图
                ax5 = plt.subplot(425, projection='3d')
                ax5.plot(self.tremor_x[-1],self.tremor_y[-1],self.tremor_z[-1],
                         linewidth=0.3)
                plt.savefig(self.image_path + "/tremor_move/" + str(tremor_file_name_index).zfill(7) + ".png")



        # speed 计算有些过于慢了 存一下
        #self.average_speed = np.array(self.average_speed)
        #np.save("./Intermediate_results/tremor_average_speed.npy", self.average_speed)

    # 排序，并将文件保存为一个文件名列表，跑一次就不用再管了
    def sort_amp(self):
        t_a = []
        t_m = []
        for tremor_file_name_index, tremor_file_name in enumerate(self.address):
            cross_zero = 0
            # 验证，分割，读取，幅度变化
            # 分割
            tremor_file_name_detail = tremor_file_name[0].split('/')
            tremor_capid = tremor_file_name_detail[8]
            # 手工筛选
            # if tremor_capid not in ["T034_Left","T033_Left"]:#"T002_Left","T040_Right","T055_Left","T020_Left","T008_Right","T001_Left"
            #    print("pass "+tremor_file_name[0] )
            #    continue
            tremor_seqid = tremor_file_name_detail[9]
            # 验证
            exame_id = 0
            for i in [0, 1, 2, 4, 5, 6]:
                exam_detail = tremor_file_name[exame_id].split('/')
                exam_capid = exam_detail[8]
                exam_seqid = exam_detail[9]
                exam_xyzid = exam_detail[10]
                if exam_capid != tremor_capid or exam_seqid != tremor_seqid:
                    raise RuntimeError("验证文件名时出错")
                if exam_xyzid != str(i).zfill(2) + ".txt":
                    raise RuntimeError("验证文件名时出错")
                exame_id += 1
            # 读取
            tremor_x = np.loadtxt(tremor_file_name[0])
            tremor_y = np.loadtxt(tremor_file_name[1])  # y-axis pointing towards the fingers.
            tremor_z = np.loadtxt(tremor_file_name[2])  # z-axis pointing away from the skin surface
            # 幅度统计
            tremor_xyz = np.sqrt(tremor_x ** 2 + tremor_y ** 2 + tremor_z ** 2)
            t_a.append(np.mean(tremor_xyz))
            t_m.append(np.max(tremor_xyz))
        t_a_sort = np.argsort(t_a)
        np.save("t_a_sort.npy", t_a_sort)
        tremor_file_name_all = self.address[t_a_sort]
        np.save("tremor_file_name_all.npy", tremor_file_name_all)
volunter_motion_1 = volunter_motion(address_="/home/chen/Documents/.git/Mocap_SIG18_Data/training_data",
                                    save_file_="volunter_motuion1.pkl",
                                    image_path_="/home/chen/Documents/GNN_hand_research/data_generate/analysis_image",
                                    to_cm_rate_ = 1.0)
tremor_motion_1 = tremor_motion(address_="/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/extract_data",
                                save_file_="tremor_motuion1.pkl",
                                image_path_="/home/chen/Documents/GNN_hand_research/data_generate/analysis_image",
                                to_cm_rate_=300000.0)

# 清道夫的工作
if not os.path.exists(tremor_motion_1.image_path + "/final_mating/"):
    os.makedirs(tremor_motion_1.image_path + "/final_mating/")
    print('create ' + tremor_motion_1.image_path + "/final_mating/")
volunter_motion.del_file(tremor_motion_1.image_path + "/final_mating/")

#查看平均幅度范围，确定振幅数量100，计算步长，定义一个振幅列表
amp_max = tremor_motion_1.tremor_amp_avg[-1]
amp_range = []
amp_num = 100
for amp_i in range(amp_num+1):
    amp_range.append(amp_max*amp_i/amp_num)
#遍历振幅
#print(tremor_motion_1.tremor_amp_avg)
#print(amp_range)


for amp_index in tqdm(range(amp_num)):

    # 清道夫的工作
    db_path = "/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/degree_dataset/2/" + str(amp_index).zfill(3)+"/"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print('create ' + db_path)
    volunter_motion.del_file(db_path)

    isfake = False
    amp_up = amp_range[amp_index+1]
    amp_down = amp_range[amp_index]
    #寻找满足幅度约束的id列表
    tremor_in_list = []
    for one_tremor_amp_i, one_tremor_amp in enumerate(tremor_motion_1.tremor_amp_avg):
        if one_tremor_amp<=amp_up and one_tremor_amp>=amp_down:
            tremor_in_list.append(one_tremor_amp_i)
    # 如果有空的，就随机找一个，计算放大倍数范围，随机放大
    if not tremor_in_list:
        isfake = True
        rand_id = random.randint(0,len(tremor_motion_1.tremor_amp_avg)-1)
        rand_amp_ = tremor_motion_1.tremor_amp_avg[rand_id]
        rand_amp = random.uniform(amp_down/rand_amp_, amp_up/rand_amp_)
        #print("rand ID: "+str(rand_id)+"  "+"rand_amp"+str(rand_amp_)+"  "+"Faking rate"+str(rand_amp))
        tremor_in_list.append(rand_id)
    #随机选择符合这个振幅范围的一个震荡文件
    choose_tremor_file = tremor_in_list[random.randint(0, len(tremor_in_list)-1)]
    #print(choose_tremor_file)
    choose_tremor_x = np.array(tremor_motion_1.tremor_x[choose_tremor_file])
    choose_tremor_y = np.array(tremor_motion_1.tremor_y[choose_tremor_file])
    choose_tremor_z = np.array(tremor_motion_1.tremor_z[choose_tremor_file])
    #choose_tremor_speed = tremor_motion_1.average_speed[choose_tremor_file * 10]
    if isfake:
        choose_tremor_x_ = np.array(choose_tremor_x)
        choose_tremor_y_ = np.array(choose_tremor_y)
        choose_tremor_z_ = np.array(choose_tremor_z)
        choose_tremor_x *= rand_amp
        choose_tremor_y *= rand_amp
        choose_tremor_z *= rand_amp
    #随机选择主动运动的文件,之后改成遍历
    #choose_volunter_file = random.randint(0, len(volunter_motion_1.pose_all)-1)
    for volunter_motion_data_i, volunter_motion_data in enumerate(volunter_motion_1.pose_all):
        #volunter_motion_data = volunter_motion_1.pose_all[choose_volunter_file]
        #choose_volunter_speed = volunter_motion_1.average_speed[choose_tremor_file * 10]
        #遍历速度比, 帧率是500倍的差距
        #计算速度比，对主观运动进行插值
        #我们把震颤运动降采样到100hz（兼顾估计的频率和对震颤的信号保持），为了维持现在的速度比，我们研究过volunter的采样率应该在20hz，至少应该插值到100hz
        #如果插值地更多的话，那么相当于减速了
        vol_amp_num = 10
        volunter_motion_data_x = np.linspace(0, 1, np.size(volunter_motion_data, axis=0))
        volunter_motion_data_xa = np.linspace(0, 1, vol_amp_num*np.size(volunter_motion_data, axis=0))
        volunter_motion_data_append = np.zeros([vol_amp_num*np.size(volunter_motion_data, axis=0),np.size(volunter_motion_data, axis=1)])
        for volunter_motion_data_i2 in range(np.size(volunter_motion_data, axis=1)):
            fun_interp = interpolate.interp1d(volunter_motion_data_x, volunter_motion_data[:,volunter_motion_data_i2], kind='cubic')
            volunter_motion_data_append[:,volunter_motion_data_i2] = fun_interp(volunter_motion_data_xa)

        # 时间匹配
        need_rep = True
        tremor_time = np.size(choose_tremor_x) / 1000.0
        pose_time = np.size(volunter_motion_data_append, 0) / 100.0
        if pose_time > tremor_time:
            #print("pose_time is: " + str(pose_time) + "tremor_time is:" + str(tremor_time))
            need_rep = False
        if need_rep:     #重复
            repeat_time = tremor_time / pose_time
            repeat_time = int(math.floor(repeat_time))

            # pose
            onefile_data_reverse = np.zeros((np.size(volunter_motion_data_append, 0), np.size(volunter_motion_data_append, 1)))
            for trverse_i in range(np.size(volunter_motion_data_append, 0)):
                onefile_data_reverse[trverse_i, :] = volunter_motion_data_append[np.size(volunter_motion_data_append, 0) - trverse_i - 1, :]

            onefile_data_repeat = volunter_motion_data_append
            for repeat_time_i in range(repeat_time - 1):
                if repeat_time_i % 2 == 0:
                    onefile_data_repeat = np.concatenate((onefile_data_repeat, onefile_data_reverse), axis=0)
                else:
                    onefile_data_repeat = np.concatenate((onefile_data_repeat, volunter_motion_data_append), axis=0)

            # 坐标轴
            onefile_xyz_axis = volunter_motion.compute_one_axis(volunter_motion_data_append)
            onefile_xyz_axis_reverse = np.zeros((np.size(onefile_xyz_axis, 0), np.size(onefile_xyz_axis, 1)))
            for trverse_i in range(np.size(onefile_xyz_axis, 0)):
                onefile_xyz_axis_reverse[trverse_i, :] = onefile_xyz_axis[np.size(onefile_xyz_axis, 0) - trverse_i - 1, :]

            onefile_xyz_axis_repeat = onefile_xyz_axis
            for repeat_time_i in range(repeat_time - 1):
                if repeat_time_i % 2 == 0:
                    onefile_xyz_axis_repeat = np.concatenate((onefile_xyz_axis_repeat, onefile_xyz_axis_reverse), axis=0)
                else:
                    onefile_xyz_axis_repeat = np.concatenate((onefile_xyz_axis_repeat, onefile_xyz_axis), axis=0)
        else:               #截取

            vol_cut = int(math.floor(np.size(choose_tremor_x) / 10.0))

            onefile_data_repeat = volunter_motion_data_append[:vol_cut,:]
            onefile_xyz_axis_repeat = volunter_motion.compute_one_axis(volunter_motion_data_append)[:vol_cut,:]

        # 采样tremor
        sampl_interval = 10
        if np.size(onefile_data_repeat, 0) * sampl_interval > np.size(choose_tremor_x):
            raise RuntimeError("too much sample point, which is wired")
        onefile_data_repeat_ampx = np.zeros(np.size(onefile_data_repeat, 0))
        onefile_data_repeat_ampy = np.zeros(np.size(onefile_data_repeat, 0))
        onefile_data_repeat_ampz = np.zeros(np.size(onefile_data_repeat, 0))
        for amp_i in range(np.size(onefile_data_repeat, 0)):
            onefile_data_repeat_ampx[amp_i] = choose_tremor_x[amp_i * sampl_interval]
            onefile_data_repeat_ampy[amp_i] = choose_tremor_y[amp_i * sampl_interval]
            onefile_data_repeat_ampz[amp_i] = choose_tremor_z[amp_i * sampl_interval]

        # 将加速度计的坐标轴的三个振幅x'y'z'，映射到世界坐标系上(xyz)(xyz)(xyz)
        for mapping_i in range(3):
            onefile_xyz_axis_repeat[:, mapping_i] *= onefile_data_repeat_ampx
            onefile_xyz_axis_repeat[:, mapping_i + 3] *= onefile_data_repeat_ampy
            onefile_xyz_axis_repeat[:, mapping_i + 6] *= onefile_data_repeat_ampz

        onefile_xyz_axis_repeat[:, 0] = onefile_xyz_axis_repeat[:, 0] + onefile_xyz_axis_repeat[:,
                                                                        3] + onefile_xyz_axis_repeat[:, 6]
        onefile_xyz_axis_repeat[:, 1] = onefile_xyz_axis_repeat[:, 1] + onefile_xyz_axis_repeat[:,
                                                                        4] + onefile_xyz_axis_repeat[:, 7]
        onefile_xyz_axis_repeat[:, 2] = onefile_xyz_axis_repeat[:, 2] + onefile_xyz_axis_repeat[:,
                                                                        5] + onefile_xyz_axis_repeat[:, 8]

        # 交配 onefile_data_repeat(x,9)  tremor_x(x),根据可视化效果看看是否需要调整坐标系匹配
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
        np.save(db_path + str(volunter_motion_data_i).zfill(7)+".npy", onefile_data_repeat)
        # 绘制3D图
        fig = plt.figure(1)
        fig.clear()
        ax1 = plt.subplot(331, projection='3d')
        ax2 = plt.subplot(332, projection='3d')
        ax3 = plt.subplot(333, projection='3d')
        ax1.plot(onefile_data_repeat_saved[:, 0], onefile_data_repeat_saved[:, 1], onefile_data_repeat_saved[:, 2],
                 linewidth=0.3)
        ax2.plot(onefile_xyz_axis_repeat[:, 0], onefile_xyz_axis_repeat[:, 1], onefile_xyz_axis_repeat[:, 2], linewidth=0.3)
        ax3.plot(onefile_data_repeat[:, 0], onefile_data_repeat[:, 1], onefile_data_repeat[:, 2], linewidth=0.3)
        if need_rep:
            ax4 = plt.subplot(334, projection='3d')
            ax5 = plt.subplot(335, projection='3d')
            ax6 = plt.subplot(336, projection='3d')
            ax4.plot(onefile_data_repeat_saved[:int(pose_time*100), 0], onefile_data_repeat_saved[:int(pose_time*100), 1], onefile_data_repeat_saved[:int(pose_time*100), 2],
                     linewidth=0.3)
            ax5.plot(onefile_xyz_axis_repeat[:int(pose_time*100), 0], onefile_xyz_axis_repeat[:int(pose_time*100), 1], onefile_xyz_axis_repeat[:int(pose_time*100), 2],
                     linewidth=0.3)
            ax6.plot(onefile_data_repeat[:int(pose_time*100), 0], onefile_data_repeat[:int(pose_time*100), 1], onefile_data_repeat[:int(pose_time*100), 2], linewidth=0.3)
        if isfake:
            ax7 = plt.subplot(337, projection='3d')
            ax7.plot(choose_tremor_x_,choose_tremor_y_,choose_tremor_z_,
                     linewidth=0.3)
        plt.savefig(db_path + str(volunter_motion_data_i).zfill(7) + ".png")
