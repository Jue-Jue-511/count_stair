from sklearn.preprocessing import MinMaxScaler
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

colorizer = rs.colorizer()
align_to = rs.stream.color
align = rs.align(align_to)
mm = MinMaxScaler()
def cal_stair(depth_image):
    h_480,w_848  = depth_image.shape                                                            #图片尺寸
    w_424 = w_848//2                                                                            #横向中心点

    intr_l = np.array([[0.00165, 0, -0.69736842], [0, 0.00165, -0.39473684], [0, 0, 1]])
    for i in range(h_480):                                                                      #补齐0值   
        depth_image_i = depth_image[i, :]                                                       
        depth_image_mean = depth_image_i[depth_image_i > 0]
        depth_image_i[depth_image_i == 0] = depth_image_mean.mean()                             #用行均值填充缺失值
        depth_image[i, :] = depth_image_i
#下面是将数据标准化，因为灰度图显示必须是在0-255之间，深度值完全不在这个范围里面 一米以上就是1000+，所以将其标准化，投影到0-1之间
    data = mm.fit_transform(depth_image)                                                        
    data1 = data * 255                                                                          #投影到0-255，方便灰度图显示

    depth_image_norm = data1.astype(np.uint8)

    edges = cv2.Sobel(depth_image_norm, cv2.CV_64F, 0, 1, ksize=3)                              #边缘检测

    edges = edges.astype(np.uint8)
#此步有优化的地方，现在是完全模糊，理想的情况下是仅有横坐标的模糊处理比如运动模糊，暂时还没做。   
    img_2 = cv2.GaussianBlur(edges, (9, 9), 0)                                                  #将边缘图模糊处理，（去掉细碎的干扰）
    
    img_2[img_2 >= 128] = 255                                                                   #手动把边界分清楚（横向）此时还是848*480
    img_2[img_2 < 128] = 0
    img_2_mean = img_2.mean(axis=1)                                                             #横向求均值，分隔出来到底是楼梯层高还是楼梯宽度。
    img_2_mean[img_2_mean >= 128] = 255                                                         #手动把边界分清楚（纵向）此时是480*1
    img_2_mean[img_2_mean < 128] = 0
    len_sat = []
    len_end = []
    for i in range(len(img_2_mean) - 1):                                                        #开始检测楼梯的开始和终止
        if img_2_mean[i] == 0 and img_2_mean[i + 1] == 255:
            len_end.append(i)
        if img_2_mean[i] == 255 and img_2_mean[i + 1] == 0:
            len_sat.append(i)

    if len(len_end) > 0 and len(len_sat) > 0 and len_end[0] < len_sat[0]:                       #删除残缺的楼层，（最前面和最后面）
        del (len_end[0])
    if len(len_end) > 0 and len(len_sat) > 0 and len_end[-1] < len_sat[-1]:
        del (len_sat[-1])

    height = []
    # w,h = depth_image.shape
    if len(len_sat) == len(len_end):                                                            #对确定好的楼层  计算层高
        num = min(len(len_sat), len(len_end))
        for i in range(len(len_sat)):
            y1 = len_sat[i]
            y2 = len_end[i]
            z1 = depth_image[y1, w_424]
            z_1 = int(z1)
            camera_1 = list(z1 * np.dot(intr_l, np.array([w_424, y1, 1])))
            z2 = depth_image[y2, w_424]
            z_2 = int(z2)
            camera_2 = list(z2 * np.dot(intr_l, np.array([w_424, y2, 1])))
            z = z_1 - z_2

            len_x = camera_2[0] - camera_1[0]
            len_y = camera_2[1] - camera_1[1]
            len1 = (len_x ** 2 + len_y ** 2 + z ** 2) ** 0.5
            if len1 > 100:
                # cv2.line(color_image, (1, len_sat[i]), (847, len_sat[i]), (0, 255, 0), 2)
                # cv2.line(color_image, (1, len_end[i]), (847, len_end[i]), (0, 0, 255), 2)
                height.append(0.1 * len1)
    return num , height
