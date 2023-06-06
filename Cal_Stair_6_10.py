import time
from PIL import Image
# 可以在bag文件上运行
import datetime
import glob
# First import library
from sklearn import preprocessing

import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path


def min2(xs_ys, zs):
    N_POINTS = 10
    TARGET_X_SLOPE = 2
    TARGET_y_SLOPE = 3
    TARGET_OFFSET = 5
    EXTENTS = 5
    NOISE = 5
    '''
    # create random data
    xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
    ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
    zs = []
    for i in range(N_POINTS):
        zs.append(xs[i]*TARGET_X_SLOPE + \
                  ys[i]*TARGET_y_SLOPE + \
                  TARGET_OFFSET + np.random.normal(scale=NOISE))

    # plot raw data
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')
    '''
    mena_ = zs.mean(axis=0)
    vet_ones = np.ones((1, len(zs)))
    xs_ys = xs_ys
    A = np.concatenate((xs_ys, vet_ones), axis=0)
    #uuu = np.concatenate((xs_ys, zs), axis=0)
    # do fit
    # tmp_A = []
    # tmp_b = []
    # for i in range(len(xs)):
    #     tmp_A.append([xs[i], ys[i], 1])
    #     tmp_b.append(zs[i])
    b = np.matrix(zs).T  # b:(14990,1)
    A = np.matrix(A).T  # A:(14990,3)
    # print(A.shape)
    # print('lens:',len(zs))

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    onee = residual/mena_*1000
    #aaa = np.linalg.svd(A)
    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)
    # print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    return fit[0], fit[1], fit[2], residual / len(zs),onee/ len(zs)
def angle_and_height(a1, a2, a3, b1, b2, b3):
    u_up = [a1, a2, -1]
    u_up = np.array(u_up)
    u_down = [b1, b2, -1]
    u_down = np.array(u_down)

    len_u_up = np.sqrt(u_up.dot(u_up))
    len_u_down = np.sqrt(u_down.dot(u_down))

    cos_angle_up = u_up.dot(u_down) / (len_u_down * len_u_up)
    if 1 > cos_angle_up > -1:
        angle_up = int(np.arccos(cos_angle_up) * 180 / 3.1415926)
        if angle_up > 90:
            angle_up = 180 - angle_up
    else:
        print('up wrong', cos_angle_up)

    d_all_up = np.sqrt(b1 ** 2 + b2 ** 2 + 1)
    d_up = b3 - a3
    d_real_up = d_up // (d_all_up * 10)
    return angle_up, d_real_up


def cal_angle_height(depth_image, color_image):
    # cv2.rectangle(color_image, (375, 0), (475, 39), (0, 255, 0), 2)          #(x1,y1),(x2,y2)
    # cv2.rectangle(color_image, (375, 40), (475, 79), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 80), (475, 119), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 120), (475, 159), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 160), (475, 199), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 200), (475, 239), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 240), (475, 279), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 280), (475, 319), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 320), (475, 359), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 360), (475, 399), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 400), (475, 439), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (375, 440), (475, 479), (0, 255, 0), 2)
    #
    # cv2.rectangle(color_image, (400, 200), (450, 250), (0, 255, 0), 2)
    #
    # cv2.rectangle(color_image, (200, 300), (500, 400), (0, 255, 0), 2)
    # cv2.rectangle(color_image, (100, 300), (150, 400), (0, 255, 0), 2)
    #
    # cv2.rectangle(depth_color_image, (200, 150), (500, 250), (255, 255, 255), 2)
    # cv2.rectangle(depth_color_image, (200, 300), (500, 400), (255, 255, 255), 2)
    # cv2.rectangle(depth_color_image, (100, 300), (150, 400), (255, 255, 255), 2)
    axis_848 = np.arange(0, 848)
    axis_double = np.vstack((axis_848, axis_848))
    axis_x = np.repeat(axis_double, 240, axis=0)

    axis_480 = np.arange(0, 480)
    axis_480 = axis_480.reshape(-1, 1)
    axis_y = np.repeat(axis_480, 848, axis=1)

    axis_x = axis_x * 0.001644656 - 0.693573169
    axis_y = axis_y * 0.001646016 - 0.407453512
    axis_z = np.ones((480, 848))
    axis_x = axis_x[np.newaxis]
    axis_y = axis_y[np.newaxis]
    axis_z = axis_z[np.newaxis]
    axis = np.concatenate((axis_x, axis_y, axis_z), axis=0)
    camera_axis = axis * depth_image
    data = np.zeros((6, 12))

    # cv2.namedWindow('color33', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('color33', color_image)
    # cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('depth', depth_image)

    up, down = 440, 479
    num = 0
    numb = []
    plane = []
    while up > 0 and down < 480:
        y_start = up
        y_end = down
        cam_depth_cal = camera_axis[:, y_start:y_end, 300:500]
        cam_depth_cal = cam_depth_cal.reshape(3, -1)
        cam_depth_cal = np.delete(cam_depth_cal, np.where(cam_depth_cal == 0)[1], axis=1)
        a1, a2, a3, error,error_ = min2(cam_depth_cal[0:2, :], cam_depth_cal[2, :])
        #print(error,error_)
        cam_depth_cal_i = camera_axis[:, y_start:y_start + 30, 300:500]
        cam_depth_cal_i = cam_depth_cal_i.reshape(3, -1)
        cam_depth_cal_i = np.delete(cam_depth_cal_i, np.where(cam_depth_cal_i == 0)[1], axis=1)
        _, _, _, error_i,error_i_ = min2(cam_depth_cal_i[0:2, :], cam_depth_cal_i[2, :])
        #print(error_i,error_i_)
        if error_i_ <= 0.1 and error_ <= 0.1 :   #确保是平面
            data_i = [round(float(a1), 5), round(float(a2), 5), round(float(a3), 5), error_i, y_start, y_end]
            numb.append(np.array(data_i))
            if down - up >= 45:
                del (numb[-2])
            up -= 10  # 现有区域是平面，继续向上扩大


        if error_i_ > 0.1 or error_ > 0.1:  # 这说明就不是平面 需要重新调整up和down
            if down - up < 45:  # 本来40内有平面，慢慢向上平移
                down = down - 10
                up = up - 10
            if down - up >= 45:  # 本来40有平面 扩大了之后扩大到边缘了
                down = up -10  #
                up = down - 40
    a = 1
    numbb = []

    for i in range(len(numb)):
        upp = numb[i][4]
        downn = numb[i][5]
        up_3d = camera_axis[:, int(upp) + 5, 400:420]
        down_3d = camera_axis[:, int(downn) - 5, 400:420]
        mid = down_3d - up_3d
        lenss = mid ** 2
        lensss = np.sqrt(lenss.sum(axis=0))
        lenssss = np.mean(lensss)
        #print(lenssss)
        if lenssss > 200:
            numbb.append(numb[i])
            cv2.rectangle(color_image, (300, int(numb[i][4])), (500, int(numb[i][5])), (0, 255, 0), 2)

    for i in range(len(numbb) - 1):         #计算两级楼梯之间的高度
        cv2.putText(color_image, str(len(numbb)), (500, 479), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        # cv2.rectangle(color_image, (300, int(numbb[i][4])), (500, int(numbb[i][5])), (0, 255, 0), 2)
        # cv2.rectangle(color_image, (300, int(numbb[i + 1][4])), (500, int(numbb[i + 1][5])), (0, 255, 0), 2)
        a1, a2, a3 = numbb[i][0:3]
        b1, b2, b3 = numbb[i + 1][0:3]
        u_up = [a1, a2, -1]
        u_up = np.array(u_up)
        u_down = [b1, b2, -1]
        u_down = np.array(u_down)

        len_u_up = np.sqrt(u_up.dot(u_up))
        len_u_down = np.sqrt(u_down.dot(u_down))

        cos_angle_up = u_up.dot(u_down) / (len_u_down * len_u_up)
        if 1 > cos_angle_up > -1:
            angle_up = int(np.arccos(cos_angle_up) * 180 / 3.1415926)
            if angle_up > 90:
                angle_up = 180 - angle_up
            if angle_up > 45:
                angle_up = 90 - angle_up
            #if 14> angle_up > 8:
            cv2.putText(color_image, str(angle_up), (500, int(numbb[i + 1][5])), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (255, 255, 255), 2)
        else:
            print('up wrong', cos_angle_up)

        d_all_up = np.sqrt(b1 ** 2 + b2 ** 2 + 1)
        d_up = a3 - b3
        d_real_up = d_up // (d_all_up * 10)
        cv2.putText(color_image, str(int(d_real_up)), (310, int(numbb[i + 1][5])), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2)

    # cv2.namedWindow('color44', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('color44', color_image)
    # cv2.namedWindow('depth44', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('depth44', depth_image)
    '''
    for i in range(12):
            y_start = (11-i)*40
            y_end = (12 - i) * 40
            cam_depth_cal = camera_axis[:, y_start:y_end , 300:500]
            cam_depth_cal = cam_depth_cal.reshape(3, -1)
            cam_depth_cal = np.delete(cam_depth_cal, np.where(cam_depth_cal == 0)[1], axis=1)
            a1, a2, a3, error = min2(cam_depth_cal[0:2, :], cam_depth_cal[2, :])
            if error >0.1 and i<11:
                y_start = y_start - 20
                y_end = y_end - 20
                cam_depth_cal = camera_axis[:, y_start:y_end, 300:500]
                cam_depth_cal = cam_depth_cal.reshape(3, -1)
                cam_depth_cal = np.delete(cam_depth_cal, np.where(cam_depth_cal == 0)[1], axis=1)
                a1, a2, a3, error = min2(cam_depth_cal[0:2, :], cam_depth_cal[2, :])
            if error > 0.1 and i == 11:
                y_start = y_start + 20
                y_end = y_end + 20
                cam_depth_cal = camera_axis[:, y_start:y_end, 300:500]
                cam_depth_cal = cam_depth_cal.reshape(3, -1)
                cam_depth_cal = np.delete(cam_depth_cal, np.where(cam_depth_cal == 0)[1], axis=1)
                a1, a2, a3, error = min2(cam_depth_cal[0:2, :], cam_depth_cal[2, :])
                cv2.rectangle(color_image, (300, y_start), (500, y_end), (0, 255, 0), 2)
            data[:,i] = a1, a2, a3, error,y_start,y_end

    cam_depth_left = camera_axis[:, 240:280, 300:400]         #[y1:y2,x1:x2]
    cam_depth_left = cam_depth_left.reshape(3, -1)
    cam_depth_left = np.delete(cam_depth_left, np.where(cam_depth_left == 0)[1], axis=1)

    cam_depth_mid = camera_axis[:,240:280, 400:500]
    cam_depth_mid = cam_depth_mid.reshape(3, -1)
    cam_depth_mid = np.delete(cam_depth_mid, np.where(cam_depth_mid == 0)[1], axis=1)

    cam_depth_rig = camera_axis[:,240:280, 500:600]
    cam_depth_rig = cam_depth_rig.reshape(3, -1)
    cam_depth_rig = np.delete(cam_depth_rig, np.where(cam_depth_rig == 0)[1], axis=1)

    a1, a2, a3, error_left = min2(cam_depth_left[0:2, :], cam_depth_left[2, :])
    b1, b2, b3, error_mid = min2(cam_depth_mid[0:2, :], cam_depth_mid[2, :])
    c1, c2, c3, error_rig = min2(cam_depth_rig[0:2, :], cam_depth_rig[2, :])
    angle1,height1 = angle_and_height(a1, a2, a3,b1, b2, b3)
    angle2, height2 = angle_and_height(a1, a2, a3, c1, c2, c3)
    angle3, height3 = angle_and_height(b1, b2, b3, c1, c2, c3)
    cv2.putText(color_image, str(round(data[3,0],4)),  (500, 479), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3, 1], 4)), (500, 439), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,2], 4)), (500,399), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,3], 4)), (500,359), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,4], 4)), (500,319), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,5], 4)), (500,279), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,6], 4)), (500,239), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,7], 4)), (500,199), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,8], 4)), (500,159), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,9], 4)), (500,119), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,10], 4)), (500,79), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(color_image, str(round(data[3,11], 4)), (500,39), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # cv2.putText(color_image, str(angle2),  (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    # cv2.putText(color_image, str(angle3),  (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    # cv2.putText(color_image, str(height1),  (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    # cv2.putText(color_image, str(height2),  (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    # cv2.putText(color_image, str(height3),  (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    '''

    # u_up = [a1, a2, -1]
    # u_up = np.array(u_up)
    # u_down = [b1, b2, -1]
    # u_down = np.array(u_down)
    #
    # len_u_up = np.sqrt(u_up.dot(u_up))
    # len_u_down = np.sqrt(u_down.dot(u_down))
    #
    # cos_angle_up = u_up.dot(u_down) / (len_u_down * len_u_up)
    # if 1 > cos_angle_up > -1:
    #     angle_up = int(np.arccos(cos_angle_up) * 180 / 3.1415926)
    #     if angle_up > 90:
    #         angle_up = 180 - angle_up
    # else:
    #     print('up wrong', cos_angle_up)
    #
    # d_all_up = np.sqrt(b1 ** 2 + b2 ** 2 + 1)
    # d_up = b3 - a3
    # d_real_up = d_up // (d_all_up * 10)


    # if error_up<0.1 and error_down <0.1 and angle_up < 5 and d_real_up >5:
    #     cv2.putText(color_image, 'stair is :%s' % d_real_up, (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(255, 255, 255), 2)
    #
    # if error_up < 0.1 and error_down < 0.1 and 20>angle_up > 10 :
    #     cv2.putText(color_image, 'angle is :%s' % angle_up, (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),2)

    # cv2.putText(color_image, '%.2f' % error_up, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #             (255, 255, 255), 2)
    # cv2.putText(color_image, '%.2f' % error_down, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #             (255, 255, 255), 2)
    # cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('color', color_image)
    # cv2.waitKey(1)
    return color_image


color_file = os.path.join('/home/data/juejue/Realsense/20210724_204051/color/*.png')
depth_file = os.path.join('/home/data/juejue/Realsense/20210724_204051/depth/*.png')
#color_depth_file = os.path.join('/home/data/juejue/Realsense/20210724_204051/color_depth/*.png')
color_ = sorted(glob.glob(color_file))
depth_ = sorted(glob.glob(depth_file))
#color_depth_ = sorted(glob.glob(color_depth_file))
for i in range (len(color_)):
    name = os.path.split(color_[i])[-1]
    depth_i = os.path.join('/home/data/juejue/Realsense/20210724_204051/depth',name)
    #color_depth_i = os.path.join('/home/data/juejue/Realsense/20210724_204051/color_depth', name)
    color_image = Image.open(color_[i])
    color_image = np.array(color_image, dtype='uint8')
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    depth_image = Image.open(depth_i)
    depth_image = np.array(depth_image, dtype='uint16')

    #color_depth_image = Image.open(color_depth_i)
    #color_depth_image = np.array(color_depth_image, dtype='uint8')
    color_image = cal_angle_height(depth_image, color_image)

    color_save_path = '/home/data/juejue/Realsense/20210724_204051/finish5/' + os.path.split(color_[i])[-1]

    cv2.imwrite(color_save_path, color_image)
    # print('a')
    # cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('color', color_image)
    # cv2.waitKey(1)
    # cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('depth', color_depth_image)

    # print('a')
