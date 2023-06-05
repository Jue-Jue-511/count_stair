import numpy as np
import pyrealsense2 as rs2
import time

intr_l = np.array([[0.001637, 0, -0.69547], [0, 0.00164, -0.39473684], [0, 0, 1]])

def min2(xs, ys, zs):
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)


    # Manual solution
    fit = (A.T * A).I * A.T * b

    return fit[0], fit[1], fit[2]

def cal_angle (depth_image,mask1,mask2,min2):
    h480, w848 = depth_image.shape



    x_up, y_up, z_up = [], [], []
    for i in range(w848):
        for j in range(h480):
            # if depth_image[j, i] == 0:
            # 发现绘制的点中有很多深度几万到六万多不等的，把它们过滤掉
            x_up_ = i               #848
            y_up_ = j               #480
            point_depth = depth_image[y_up_, x_up_]
            if point_depth == 0 or mask1[y_up_,x_up_] == 0:
                continue
            else:
                dis_up = depth_image[x_up_, y_up_]
                #camera_coordinate = rs2.rs2_deproject_pixel_to_point(intrin=intr_l, pixel=[x_up_, y_up_],depth=dis_up)
                camera = list(dis_up * np.dot(intr_l, np.array([y_up_, x_up_, 1])))
                x_up.append(camera[0])
                y_up.append(camera[1])
                z_up.append(dis_up)




    x_down, y_down, z_down = [], [], []
    for i in range(w848):
        for j in range(h480):
            # if depth_image[j, i] == 0:
            # 发现绘制的点中有很多深度几万到六万多不等的，把它们过滤掉
            x_down_ = i
            y_down_ = j

            point_depth = depth_image[y_down_, x_down_]
            if point_depth == 0 or point_depth >= 10000:
                continue
            else:
                dis_down = depth_image[y_down_, x_down_]
                #camera_coordinate = rs2.rs2_deproject_pixel_to_point(intrin=intr_l, pixel=[x_down_, y_down_],depth=dis_down)
                camera = list(dis_down * np.dot(intr_l, np.array([y_down_, x_down_, 1])))
                x_down.append(camera[0])
                y_down.append(camera[1])
                z_down.append(dis_down)


    # time_end = time.clock()  # 记录结束时间
    # time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    # print(time_sum)

    a1, a2, a3 = min2(x_up, y_up, z_up)
    u_up = [a1, a2, -1]
    u_up = np.array(u_up)

    b1, b2, b3 = min2(x_down, y_down, z_down)
    u_down = [b1, b2, -1]
    u_down = np.array(u_down)

    len_u_up = np.sqrt(u_up.dot(u_up))
    len_u_down = np.sqrt(u_down.dot(u_down))

    cos_angle = u_up.dot(u_down) / (len_u_down * len_u_up)
    angle = np.arccos(cos_angle) * 180 / 3.1415926

    if angle > 90:
        angle = 180 - angle

    return angle