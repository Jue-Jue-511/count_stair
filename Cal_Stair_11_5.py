#####################################################
##                                                 ##
#####################################################

'''
新方法 矩阵对矩阵
'''
import time

#可以在bag文件上运行
import datetime

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

save_video = True
a = 0
num_up = 0
num_down = 0
# save_video = False

def cal_angle_height(depth_image,color_image):
    cv2.rectangle(color_image, (200, 150), (500, 250), (0, 255, 0), 2)
    cv2.rectangle(color_image, (200, 300), (500, 400), (0, 255, 0), 2)
    #cv2.rectangle(color_image, (100, 300), (150, 400), (0, 255, 0), 2)

    cv2.rectangle(depth_color_image, (200, 150), (500, 250), (255, 255, 255), 2)
    cv2.rectangle(depth_color_image, (200, 300), (500, 400), (255, 255, 255), 2)
    #cv2.rectangle(depth_color_image, (100, 300), (150, 400), (255, 255, 255), 2)
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

    cam_depth_up = camera_axis[:, 150:250, 200:500]
    cam_depth_up = cam_depth_up.reshape(3, -1)
    cam_depth_up = np.delete(cam_depth_up, np.where(cam_depth_up == 0)[1], axis=1)

    cam_depth_down = camera_axis[:, 300:400, 200:500]
    cam_depth_down = cam_depth_down.reshape(3, -1)
    cam_depth_down = np.delete(cam_depth_down, np.where(cam_depth_down == 0)[1], axis=1)

    a1, a2, a3, error_up = min2(cam_depth_up[0:2, :], cam_depth_up[2, :])
    b1, b2, b3, error_down = min2(cam_depth_down[0:2, :], cam_depth_down[2, :])

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

    d_all_left = np.sqrt(b1 ** 2 + b2 ** 2 + 1)
    if error_up<0.1 and error_down <0.1 and angle_up < 5 and d_real_up >5:
        cv2.putText(color_image, 'stair is :%s' % d_real_up, (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(255, 255, 255), 2)

    if error_up < 0.1 and error_down < 0.1 and 20>angle_up > 10 :
        cv2.putText(color_image, 'angle is :%s' % angle_up, (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),2)

    # cv2.putText(color_image, '%.2f' % error_up, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #             (255, 255, 255), 2)
    # cv2.putText(color_image, '%.2f' % error_down, (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
    #             (255, 255, 255), 2)
    return color_image












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
    vet_ones = np.ones((1,len(zs)))
    xs_ys = xs_ys
    A = np.concatenate((xs_ys, vet_ones), axis=0)

    # do fit
    # tmp_A = []
    # tmp_b = []
    # for i in range(len(xs)):
    #     tmp_A.append([xs[i], ys[i], 1])
    #     tmp_b.append(zs[i])
    b = np.matrix(zs).T              #b:(14990,1)
    A =  np.matrix(A).T                #A:(14990,3)
    print(A.shape)
    print('lens:',len(zs))

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)
    #print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    return fit[0], fit[1], fit[2],residual/len(zs)


try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, '/home/data/JueJue/Realsense/20210724_204051.bag')
    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth,848,480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)  # 打开文件一定是要rgb8
    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in

    # Create colorizer object
    colorizer = rs.colorizer()
    align_to = rs.stream.color
    align = rs.align(align_to)


    dis_list = []
    intr_l = np.array([[0.001644656, 0, -0.693573169], [0, 0.001646016, -0.407453512], [0, 0, 1]])

    # Streaming loop
    while True:
        # for x in range(50):
        #     #pipe.wait_for_frames()
        #     frames = pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()

        # 将深度框与颜色框对齐
        aligned_frames = align.process(frames)

        # 获取对齐的帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_color_frame = colorizer.colorize(aligned_depth_frame)
        # 验证两个帧是否有效
        if not aligned_depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        h480, w848 = depth_image.shape

        start5 = time.time()




        cv2.namedWindow('color2', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('color2', color_image)

        depth_profile = aligned_depth_frame.get_profile()
        dvsprofile = rs.video_stream_profile(depth_profile)
        intrin = dvsprofile.get_intrinsics()
        # print(intrin.ppx)
        # mask_up = np.zeros((h480, w848), dtype=int)
        # mask_up[50:150, 50:150] = 1
        #
        # mask_down = np.zeros((h480, w848), dtype=int)
        # mask_down[350:450, 350:450] = 1





        depth_image_up = depth_image[150:200, 200:500]
        h480_up, w848_up = depth_image_up.shape
        depth_image_down = depth_image[300:400, 200:500]
        h480_down, w848_down = depth_image_down.shape
        depth_image_left = depth_image[300:400, 100:150]
        h480_left, w848_left = depth_image_left.shape





        '''
        axis_x_up = axis_x[150:200, 200:500]
        _x_up = axis_x_up.reshape(1, -1)
        _x_up = _x_up.astype(float).tolist()
        axis_y_up = axis_y[150:200, 200:500]
        _y_up = axis_y_up.reshape(1, -1)
        _y_up = _y_up.astype(float).tolist()
        axis_z_up = axis_z[150:200, 200:500]
        _z_up = axis_z_up.reshape(1, -1)
        _z_up = _z_up.astype(float).tolist()

        axis_x_down = axis_x[250:350, 200:500]
        _x_down = axis_x_down.reshape(1,-1)
        _x_down = _x_down.astype(float).tolist()
        axis_y_down = axis_y[250:350, 200:500]
        _y_down = axis_y_down.reshape(1,-1)
        _y_down = _y_down.astype(float).tolist()
        axis_z_down = axis_z[250:350, 200:500]
        _z_down = axis_z_down.reshape(1,-1)
        _z_down = _z_down.astype(float).tolist()

        axis_x_left = axis_x[300:400, 100:150]
        _x_left = axis_x_left.reshape(1, -1)
        _x_left = _x_left.astype(float).tolist()
        axis_y_left = axis_y[300:400, 100:150]
        _y_left = axis_y_left.reshape(1, -1)
        _y_left = _y_left.astype(float).tolist()
        axis_z_left = axis_z[300:400, 100:150]
        _z_left = axis_z_left.reshape(1, -1)
        _z_left = _z_left.astype(float).tolist()
        '''
        # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()
        # print(depth_intrin)
        # intr = profile.as_video_stream_profile().get_intrinsics()

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        '''
        x_, y_, z_ = [], [], []
        # for n in range(8)
        for i in range(w848):
            for j in range(h480):
                # if depth_image[j, i] == 0:
                # 发现绘制的点中有很多深度几万到六万多不等的，把它们过滤掉
                x__ = i
                y__ = j
                point_depth = depth_image[y__, x__]
                if point_depth == 0 or point_depth >= 10000:
                    continue
                else:
                    dis_down = aligned_depth_frame.get_distance(x__, y__)
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x__, y__],
                                                                        depth=dis_down)
                    camera_2 = dis_down * np.dot(intr_l, np.array([x__, y__, 1]))
                    x_.append(camera_2[0] * 1000)
                    y_.append(camera_2[1] * 1000)
                    z_.append(camera_2[2] * 1000)
        '''
        '''
        end = time.time()
        print(end - start)
        # angle = cal_angle(depth_image, mask_up, mask_down, min2)
        '''


        '''
        x_up, y_up, z_up = [], [], []
        X_up, Y_up, Z_up = [], [], []
        x_up_1, y_up_1, z_up_1 = [], [], []
        for i in range(w848_up):
            for j in range(h480_up):
                # if depth_image[j, i] == 0:
                # 发现绘制的点中有很多深度几万到六万多不等的，把它们过滤掉
                x_up_ = i + 200
                y_up_ = j + 150
                point_depth = depth_image[y_up_, x_up_]
                if point_depth == 0 or point_depth >= 10000:
                    continue
                else:
                    dis_up = aligned_depth_frame.get_distance(x_up_, y_up_)
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x_up_, y_up_],
                                                                        depth=dis_up)
                    camera_coordinate = np.array(camera_coordinate)
                    camera_up = list(dis_up * np.dot(intr_l, np.array([x_up_, y_up_, 1])))
                    # X_up.append(x_up_)
                    # Y_up.append(y_up_)
                    # Z_up.append(camera_coordinate_1[2])
                    x_up.append(camera_up[0] * 1000)
                    y_up.append(camera_up[1] * 1000)
                    z_up.append(camera_up[2]*1000)
        # X_up = np.mat(X_up)
        # Y_up = np.mat(Y_up)
        # Z_up = np.mat(Z_up)
        # x_up = np.mat(x_up)
        # y_up = np.mat(y_up)
        # z_up = np.mat(z_up)
        # XYZ = np.concatenate((X_up, Y_up, Z_up), axis=0)
        # XYZ = XYZ.T
        # xyz = np.concatenate((x_up, y_up, z_up), axis=0)
        # xyz = xyz.T
        #             x_up_1.append(camera_up[0] * 1000)
        #             y_up_1.append(camera_up[1] * 1000)
        #             z_up_1.append(camera_up[2] * 1000)
        #np.linalg.solve(xyz, XYZ)
        end1 = time.time()
        print('x_up:time',end1 - start1)
        # 将数据写入numpy文件
        # count += 1
        # np.savez('{}_{}'.format(count, time.time()), x=x_up, y=y_up, z=z_up)

        x_down, y_down, z_down = [], [], []
        for i in range(w848_down):
            for j in range(h480_down):
                # if depth_image[j, i] == 0:
                # 发现绘制的点中有很多深度几万到六万多不等的，把它们过滤掉
                x_down_ = i + 200
                y_down_ = j + 300
                point_depth = depth_image[y_down_, x_down_]
                if point_depth == 0 or point_depth >= 10000:
                    continue
                else:
                    dis_down = aligned_depth_frame.get_distance(x_down_, y_down_)
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x_down_, y_down_],
                                                                        depth=dis_down)
                    camera_2 = list(dis_down * np.dot(intr_l, np.array([x_down_, y_down_, 1])))
                    x_down.append(camera_2[0] * 1000)
                    y_down.append(camera_2[1] * 1000)
                    z_down.append(camera_2[2] * 1000)
        

        x_left, y_left, z_left = [], [], []
        for i in range(w848_left):
            for j in range(h480_left):
                # if depth_image[j, i] == 0:
                # 发现绘制的点中有很多深度几万到六万多不等的，把它们过滤掉
                x_left_ = i +100
                y_left_ = j + 300
                point_depth = depth_image[y_left_, x_left_]
                if point_depth == 0 or point_depth >= 10000:
                    continue
                else:
                    dis_left = aligned_depth_frame.get_distance(x_left_, y_left_)
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x_left_, y_left_],
                                                                        depth=dis_left)
                    camera_2 = list(dis_left * np.dot(intr_l, np.array([x_left_, y_left_, 1])))
                    x_left.append(camera_2[0] * 1000)
                    y_left.append(camera_2[1] * 1000)
                    z_left.append(camera_2[2] * 1000)
        '''
        '''

        a1, a2, a3 = min2(_x_up[0], _y_up[0], _z_up[0])
        b1, b2, b3 = min2(_x_down[0],_y_down[0], _z_down[0])
        c1, c2, c3 = min2(_x_left[0], _y_left[0], _z_left[0])
        '''


        end5 = time.time()
        print('time5:', end5 - start5)
        #c1, c2, c3,error_left = min2(x_left, y_left, z_left)
        # 将数据写入numpy文件
        # count += 1
        # np.savez('{}_{}'.format(count, time.time()), x=x_down, y=y_down, z=z_down)
        #a1, a2, a3 = min2(x_up, y_up, z_up)
        # print('-------------------------------')



        # print("(1)----------------------------")
        #b1, b2, b3 = min2(x_down, y_down, z_down)
        # print(b1-a1,b2-a2,b3-a3)

        #c1, c2, c3 = min2(x_left, y_left, z_left)
        # print(b1-a1,b2-a2,b3-a3)
        #u_left = [c1, c2, -1]
        # print(u_down)
        # print("(2)----------------------------")


        color_image = cal_angle_height(depth_image,color_image)


        #u_left = np.array(u_left)
        # print(u_up)


        #len_u_left = np.sqrt(u_left.dot(u_left))






        # cos_angle_left = u_left.dot(u_down) / (len_u_down * len_u_left)
        # if 1 > cos_angle_left > -1:
        #     angle_left  = int(np.arccos(cos_angle_left ) * 180 / 3.1415926)
        #     if angle_left  > 90:
        #         angle_left  = 180 - angle_left
        # else:
        #     print('left  wrong', cos_angle_left )

        # if angle >45 :
        #     angle = 90 - angle

        #d_left = b3 - c3
        #d_real_left = d_left // (d_all_up * 10)

        #if error_up < 900 and error_down <1500 and angle_up < 30 and d_real_up > 4 :




        # if error_left < 900 and error_down < 1500:
        #     cv2.putText(color_image, 'angle is %s' % angle_left, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        #                 (255, 255, 255), 2)


        # cv2.putText(color_image, '%.2f' % error_left, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        #             (255, 255, 255), 2)
        #cv2.putText(color_image, 'stair is :%s' % d_real_up, (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),2)
        #cv2.putText(color_image, 'angle is %s' % angle_up, (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        #
        # cv2.putText(color_image, 'stair is :%s' % d_real_left, (150,350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),2)
        # cv2.putText(color_image, 'angle is %s' % angle_left, (150,300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        #cv2.putText(color_image, 'stair is :%s' % d_real, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('depth', depth_color_image)
        cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('color', color_image)
        a += 1
        '''
        if save_video:
            if cv2.waitKey(1) & 0xFF != ord('q'):
                fps, w, h = 15, color_image.shape[1], color_image.shape[0]

                if a == 1:
                    vid_writer = cv2.VideoWriter("{}.mp4".format(datetime.datetime.now().strftime('%H%M%S')),
                                                 cv2.VideoWriter_fourcc(*'mp4v'), 2, (w, h))
                vid_writer.write(color_image)
                # print("\r video [{}/{}] save path{} ".format(a,300,save_vidio),end="")
            else:
                print(" Done,vidio save to{}".format("./test.mp4"))
                # vid_writer.release()
                cv2.destroyAllWindows()
                break
        '''
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
finally:
    pass