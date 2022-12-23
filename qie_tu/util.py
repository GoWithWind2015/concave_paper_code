#coding=utf-8

import os
import time
import math
from typing import Tuple

import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import skimage

# 全局数据结构
global_variable = {}
is_use_simple = False
is_use_watershed = False

cut_img_index = 0
class RgbColor:
    black = (0,0,0)
    red = (255,0,0)
    pink = (255,0,255)
    green = (0,255,0)
    blue = (0,0,255)

# 计算运行时间
class CalcRunTime:
    def __init__(self):
        self.start = 0
        self.end = 0

    def start_process(self):
        self.start = time.process_time()

    def end_process(self):
        self.end = time.process_time()
        print("程序运行时间: ", self.end - self.start, " seconds")

def skiLine(r0,c0,r1,c1):
    rr,cc = skimage.draw.line(int(r0), int(c0), int(r1),int(c1))
    points = zip(rr,cc)
    # points = [iter_v for iter_v in zip(rr,cc)]
    return points

def get_line_at_distance_d(r0,c0,r1,c1,d):
    # k != 0 或者 无穷 时候单独处理
    if r0 == r1:
        delta_x = d
        delta_y = 0
    elif c0 == c1:
        delta_x = 0
        delta_y = d
    else:
        k = (c1 - c0) / (r1 - r0)
        delta_x = (k * d) / (math.sqrt(1 + k ** 2))
        delta_y = -1/k * delta_x

    # 返回值中 po_p p1_p 构成一条直线 ,p0_n,_p1_n构成一条直线
    p0_p  = (r0 + delta_x,c0 + delta_y)
    p1_p = (r1 + delta_x,c1 + delta_y)
    p0_n = (r0 - delta_x,c0 - delta_y)
    p1_n = (r1 - delta_x,c1 - delta_y)

    return p0_p,p1_p,p0_n,p1_n
    


# 返回值: 负数- 外侧 ,正数: 内侧,0: 线上 ,measureDist=True: 返回的是具体的数值,和轮廓线最短的距离
def cv2_pointPolygonTest(contour,pt,measureDist=True):
    position = cv2.pointPolygonTest(contour, ((int)(pt[0]),int(pt[1])), measureDist) # 若点在轮廓上
    print("position: ",position)
    return position

# 获得线和多边形关系,返回在内侧,外侧,线上点的比例,不处理 多次进出轮廓线的情况,中间部分最多出轮廓线一次
def linePolygonTest(r0,c0,r1,c1,contour):
    points = skiLine(r0,c0,r1,c1)
    points_length = 0
    outside_num = 0
    inside_num = 0
    inline_num = 0

    #  array(左边负数个数,中间负数个数,右边负数个数)
    outside_num_array = np.array([0,0,0]);
    outside_num_array_index = 0
    last_position = 0 

    for point in  points:
        points_length = points_length + 1
        position = cv2_pointPolygonTest(contour, point)
        # -1点视为在曲线上
        if position < 0:
            if points_length != 1:
                if last_position >= -1 and position < -1:
                    outside_num_array_index = outside_num_array_index + 1
            if outside_num_array_index > 2:
                print("Error: 中间多次出轮廓线")
            outside_num_array[outside_num_array_index] = outside_num_array[outside_num_array_index] + 1
        elif position == 0:
            inline_num = inline_num + 1
        last_position = position

    # 如果最右边点数为0,则说明中间的点就是最右侧的点
    if outside_num_array[2] == 0 and outside_num_array[1] != 0 and last_position < 0:
        outside_num_array[2] = outside_num_array[1]
        outside_num_array[1] = 0

    outside_num = outside_num_array[0] + outside_num_array[1] + outside_num_array[2]
    inside_num = points_length - outside_num - inline_num

    print("左边,中间,右边 点的个数: ",outside_num_array[0],outside_num_array[1],outside_num_array[2])

    return outside_num / points_length,inline_num / points_length,inside_num / points_length,outside_num,inline_num,inside_num,outside_num_array[0],outside_num_array[1],outside_num_array[2]


def is_Line_sibling_all_in_contour(r0,c0,r1,c1,contour,line_distance=6):
    if line_distance == 0:
        return -1
    p0_p,p1_p,p0_n,p1_n = get_line_at_distance_d(r0,c0,r1,c1,line_distance)
    print("in positive, line_distance: ",line_distance)
    outside_p,_,inside_p,outside_num_p,inline_num_p,inside_num_p,left_negative_num_p,middle_negative_num_p,right_negative_num_p = linePolygonTest(*p0_p,*p1_p, contour)
    print("outside_p: ",outside_p,"inside_p: ",inside_p)
    print("in negative, line_distance: ",line_distance)
    outside_n,_,inside_n,outside_num_n,inline_num_n,inside_num_n,left_negative_num_n,middle_negative_num_n,right_negative_num_n = linePolygonTest(*p0_n,*p1_n, contour)
    print("outside_n: ",outside_n,"inside_n: ",inside_n)
    is_sibling_all_in_contour = False
    # 只要line_distance !=0 就不存在在线上的情况
    # if inside_p > outside_p and inside_n > outside_n:
    outside_threshold = 0.2
    #最多不超过5个像素在外面
    outside_point_num_threshold = 6
    if (outside_p < outside_threshold and outside_num_p < outside_point_num_threshold) and (outside_n < outside_threshold or outside_num_n < outside_point_num_threshold ) and (middle_negative_num_p == 0 and middle_negative_num_n == 0):
        is_sibling_all_in_contour = True
    return is_sibling_all_in_contour
    
    
    

# 绘制直方图
def plot_hist_from_np_array(img_np_maxtrix, bins=128, option_dict=None):
    img_np_maxtrix_flatten = img_np_maxtrix.flatten()
    # bins 直方图的柱数
    pylab.hist(img_np_maxtrix_flatten, bins)
    if option_dict != None:
        if "hist_title" in option_dict:
            pylab.title(option_dict.get("hist_title"))
        if "savefig" in option_dict:
            pylab.savefig(option_dict.get("savefig"))

    # if(savefig != None):
    #     pylab.savefig('')

    pylab.show()


def plot_hist_from_PIL_Image(pil_Image, bins=128):
    img_np = np.array(pil_Image)
    plot_hist_from_np_array(img_np)


def plot_hist_from_array_list(array_list, option_dict=None):
    np_array_list = np.array(array_list)
    plot_hist_from_np_array(np_array_list, 256, option_dict)


def plot_hist_from_np_array_in_range(img_np, lowb, highb, option_dict=None):
    array_list = []
    row_start = 0
    column_start = 0
    row_end = img_np.shape[0]
    column_end = img_np.shape[1]

    if option_dict != None:
        if "plot_half" in option_dict:
            half_option = option_dict.get("plot_half")
            print(half_option)

            middle_column = int(column_end / 2)

            if half_option == "left":
                column_end = middle_column

            if half_option == "right":
                column_start = middle_column

    for i in range(row_start, row_end):
        for j in range(column_start, column_end):
            if lowb <= img_np[i][j] <= highb:
                array_list.append(img_np[i][j])
    plot_hist_from_array_list(array_list, option_dict)
    print("data number between lowb and highb: ", len(array_list))


def plot_3d_img_from_matrix(matrix):
    matrix_shape = matrix.shape
    # 横纵坐标
    x = np.arange(0, matrix_shape[1], 1)
    y = np.arange(0, matrix_shape[0], 1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(x, y, matrix)
    # ax.plot_surface(x, y, matrix, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
    plt.show()


# 获取图像信息
def get_PIL_Image_info(pil_Image: Image):
    return {
        "format": pil_Image.format,
        "info": pil_Image.info,
        "mode": pil_Image.mode,
        "size": pil_Image.size,
    }


# 对目录中的每个文件执行func
def exec_fun_for_files_in_dir(dir_name, func):
    files_in_dir = os.listdir(dir_name)
    for file in files_in_dir:
        file_abs_path = os.path.join(dir_name, file)
        func(file_abs_path)


def exec_fun_for_files_in_dir(dir_name, func):
    files_in_dir = os.listdir(dir_name)
    for file in files_in_dir:
        file_abs_path = os.path.join(dir_name, file)
        func(file_abs_path)


def print_file_name(file_base_path):
    print(file_base_path)


# option_dict = {"hist_title": img_name,
#                # "savefig": img_name.split(".")[0] + "_whole_hist.png",
#                # "savefig": img_name.split(".")[0] + "_left_hist.png",
#                "savefig": img_name.split(".")[0] + "_right_hist.png",
#                # "plot_half": "left",
#                "plot_half": "right",
#                }
# option_dict = {"hist_title": img_name}
class FunctionForFile:
    def __init__(
        self,
        hist_title=None,
        savefig=None,
        plot_half=None,
        start=1,
        background_row_number=None,
        end=46000,
    ):
        self.option_dict = {}
        self.start = start
        self.end = end
        self.background_row_number = background_row_number

        if hist_title:
            self.option_dict["hist_title"] = hist_title
        if savefig:
            self.option_dict["savefig"] = savefig
        if plot_half:
            self.option_dict["plot_half"] = plot_half

    def plot_hist_img(self, img_name):
        pil_Image = Image.open(img_name, mode="r")
        img_np = np.array(pil_Image)

        if img_name.endswith(".tif"):
            plot_hist_from_np_array_in_range(
                img_np, self.start, self.end, self.option_dict
            )

    def plot_background(self, img_name):
        pil_Image = Image.open(img_name, mode="r")
        img_np = np.array(pil_Image)
        print(img_np.shape)
        row_start = 0
        row_end = img_np.shape[0]
        column_start = 0
        column_end = img_np.shape[1]
        array_list = []

        for i in range(row_start, row_end):
            for j in range(column_start, column_end):
                if i == self.background_row_number and img_np[i][j] != 0:
                    array_list.append(img_np[i][j])
        plt.plot(array_list)
        plt.show()


# 16位深度图像转换位8位深度图像(均为单通道)
def u16_2_u8(u16_single_channel_img) -> np.ndarray:
    return cv2.convertScaleAbs(
        cv2.normalize(u16_single_channel_img, None, 0, 255, cv2.NORM_MINMAX),
        alpha=1,
        beta=0,
    )


# 将图像分割为高能图像和低能图像
def cut_image(img_u8_SC_1, rightTrunc=6) -> Tuple[np.ndarray, np.ndarray]:
    row_num = img_u8_SC_1.shape[0]
    # column_num = img_u8.shape[1]
    img_left = img_u8_SC_1[0:row_num, 81 : 2736 - rightTrunc]
    # img_left = img_u8_SC_1[0:row_num, 81:2730]
    img_right = img_u8_SC_1[0:row_num, 2897 : 5552 - rightTrunc]  #  for picture 593
    return img_left, img_right


# 使用plt显示图像
def imshow(img_title: str, img: np.ndarray, cmap="gray", showNow=True, camp="gray"):
    if camp == "bgr":
        print("以opencv方式进行绘制")
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        camp = None  # 默认以rgb进行显示

    plt.title(img_title)
    plt.imshow(img, cmap)
    if showNow:
        plt.show()


#  上下显示两张图片
def imshow2(
    img1_title: str,
    img1,
    img2_title: str,
    img2,
    saveFile=False,
    file_name="",
    camp="gray",
):
    plt.figure()
    plt.subplot(2, 1, 1)
    imshow(img1_title, img1, showNow=False, camp="gray")
    plt.subplot(2, 1, 2)
    imshow(img2_title, img2, showNow=False, camp="gray")
    # plt.get_current_fig_manager().full_screen_toggle()

    # 要先显示后保存,否则保存的图像就是空白图像
    if saveFile:
        plt.savefig(file_name)

    plt.show()


def imshow3(
    img1_title: str,
    img1,
    img2_title: str,
    img2,
    img3_title: str,
    img3,
    saveFile=False,
    file_name="",
):
    plt.figure()
    plt.subplot(3, 1, 1)
    imshow(img1_title, img1, showNow=False)
    plt.subplot(3, 1, 2)
    imshow(img2_title, img2, showNow=False)
    plt.subplot(3, 1, 3)
    imshow(img3_title, img3, showNow=False)

    # 要先显示后保存,否则保存的图像就是空白图像
    if saveFile:
        plt.savefig(file_name)

    plt.show()


def otsu_binary(src, maxValue=255) -> Tuple[np.ndarray, float]:
    threshold, dst = cv2.threshold(
        src, 0, maxValue, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return dst, threshold


# 执行全局二值阈值化，全局阈值为 46000 / 65535 * 255 = 178
def constant_binary(
    src, threshold_constant=178, maxValue=255
) -> Tuple[np.ndarray, float]:
    _, dst = cv2.threshold(src, threshold_constant, maxValue, cv2.THRESH_BINARY_INV)
    return dst

def c3_2_c1(img_left_u8_c3):
    img_c1 = cv2.cvtColor(img_left_u8_c3,cv2.COLOR_BGR2GRAY)
    return img_c1


def c1_2_c3(img_left_u8_c1):
    img_c3 = cv2.cvtColor(img_left_u8_c1,cv2.COLOR_GRAY2BGR)
    return img_c3

# 从img_u16中获取感兴趣的区域
def getROIs(img_u16: np.ndarray, min_contour_len=20):
    # 图像分割为左右两部分
    img_left, img_right = cut_image(img_u16)

    # 阈值分割，找到封闭曲线
    # dst,_ = otsu_binary(u16_2_u8(img_left))
    dst = constant_binary(u16_2_u8(img_left), threshold_constant=150)
    #todo: 需要修改为cv2.RETR_EXTERNAL
    
    contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ROIs = []
    contours_len = len(contours)
    possible_contour_indexs = []

    for i in range(0, contours_len):
        # 过滤掉边界值元素太少的点
        if len(contours[i]) > min_contour_len:
            possible_contour_indexs.append(i)

    possible_contour_indexs_len = len(possible_contour_indexs)

    if possible_contour_indexs_len > 100:
        print("contours_len:", contours_len)
        print("posssible_contour_indexs_len: ", possible_contour_indexs_len)
        print("counters数量太多，无法判断")
        imshow("too much", dst)
        imshow("too much", img_left)
        return 0

    for i in range(0, possible_contour_indexs_len):
        # 根据封闭曲线创建mask
        contour_len = len(contours[possible_contour_indexs[i]])
        # 闭合曲线最少占用的像素点，低于min_contour_len的像素点忽略
        print(contour_len)
        mask = np.zeros(dst.shape).astype(np.uint8)
        cv2.drawContours(mask, contours, possible_contour_indexs[i], 255, -1)

        # 根据mask获得左右两边的兴趣区域
        left_roi = cv2.copyTo(img_left, mask=mask)
        right_roi = cv2.copyTo(img_right, mask=mask)
        right_roi[right_roi == 0] = 1
        one_ROI = []
        one_ROI.append(left_roi)
        one_ROI.append(right_roi)
        ROIs.append(one_ROI)

    return ROIs


# 获取nparray
def getNonZero(src) -> np.ndarray:
    return src[src.nonzero()]


def getRValueFromROI(ROI) -> np.ndarray:
    left = ROI[0]
    right = ROI[1]
    RValue: np.ndarray = cv2.divide(left, right, dtype=cv2.CV_32FC1)
    return getNonZero(RValue)


def getRValuesFromROIs(ROIs):

    for i in range(0, len(ROIs)):
        R_V = getRValueFromROI(ROIs[i])
        if i == 0:
            R_Vs = R_V
        else:
            R_Vs = np.append(R_Vs, R_V)
    return R_Vs


#  绘制np_arrry/list 曲线
def plot(title, np_array):
    plt.title(title)
    plt.plot(np_array)
    plt.show()


def plot_R_hist_from_ROI(hist_title_prefix, ROI, index=None):
    R_v = getRValueFromROI(ROI)
    if index == None:
        hist_title = hist_title_prefix
    else:
        hist_title = hist_title_prefix + "_" + str(index)
    plot_hist_from_np_array(R_v, option_dict={"hist_title": hist_title})


def plot_R_hist_from_ROIs(hist_title_prefix, ROIs):
    for i in range(0, len(ROIs)):
        plot_R_hist_from_ROI("new_test", ROIs[i], index=i)


def getROIs_from_img(img_name):
    print(img_name)
    img_16 = cv2.imread(img_name, -1)

    ROIs = getROIs(img_16)

    ROI_len = len(ROIs)
    print("ROIs数量: ", ROI_len)
    if ROI_len != 0:
        r_v = getRValuesFromROIs(ROIs)
        print("r_v长度: ", len(r_v))
        return r_v
    return None


def getRValuesInDir(dir_name):
    files_in_dir = os.listdir(dir_name)
    dir_r_v_s = []
    for file in files_in_dir:
        file_abs_path = os.path.join(dir_name, file)
        r_v_s = getROIs_from_img(file_abs_path)

        if r_v_s is not None:
            dir_r_v_s = np.append(dir_r_v_s, r_v_s)
    return dir_r_v_s


def savetxt(txtFileName, savedndarray: np.ndarray):
    np.savetxt(txtFileName, savedndarray, fmt="%f")


def loadtxt(txtFileName):
    return np.loadtxt(txtFileName)


def adaptiveThresholdAndErode(
    src,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=111,
    C=20,
    erode_iterations=2,
    dilate_iterations=2,
):
    dst = cv2.adaptiveThreshold(
        src, 255, adaptiveMethod, thresholdType, blockSize, C
    )  # 适合与hard1
    # imshow("only adaptive",dst)
    dst = cv2.erode(dst, None, erode_iterations)  # 腐蚀，去除小的点
    kernel = np.ones((5, 5), np.uint8)
    dst = cv2.dilate(
        dst, kernel=kernel, iterations=dilate_iterations
    )  # 膨胀操作去除内部小的点，例如: 505_00069_30_003.tif"
    # dst = cv2.erode(dst, kernel, erode_iterations)  # 腐蚀，去除小的点
    # imshow("dst", dst)

    return dst


def cut_rectangle_from_cv2_rectangle(
    img, x, y, w, h, is_save_img=False, img_path="/tmp/temp.png"
):
    rec_img = img[y : y + h, x : x + w]
    if is_save_img:
        cv2.imwrite(img_path, rec_img)
    return rec_img


# 将file_baseName文件保存到/tmp/文件夹下
def cv2_imwrite(file_baseName, img):
    file_full_path = "/tmp/" + file_baseName + ".png"
    cv2.imwrite(file_full_path, img)


# threshold从intial_threshold开始进行阈值分割,直到有一个连通区域的面积大于给定的min_area,将符合条件的二值图像返回
# 注意返回值: return_masker , contours
def get_marker_from_min_area(
    img_u8_c1, min_area=250, intial_threshold=20, morphologyEx_iteration=2, debug=False
):
    is_meet_requirement = False
    kernel = np.ones((3, 3), np.uint8)
    erode_iteration = 3

    for i in range(intial_threshold, 255):
        if is_meet_requirement:
            break
        ret, thresh = cv2.threshold(
            img_u8_c1, intial_threshold, 255, cv.THRESH_BINARY_INV
        )
        intial_threshold = intial_threshold + 1
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.RETR_CCOMP)
        contours_len = len(contours)

        for j in range(0, contours_len):
            is_outer_contour = hierarchy[0][j][3] == -1
            contour_area = cv2.contourArea(contours[j], oriented=False)
            if is_outer_contour and contour_area > min_area:  # 是否是外轮廓，并且面积要不能低于min_area
                print(
                    "contour_len: ",
                    contours_len,
                    ", 第",
                    i,
                    "次循环,第",
                    j,
                    "个轮廓: ",
                    contour_area,
                )
                is_meet_requirement = True
                if debug:
                    # print(contours[j])
                    cv2.drawContours(img_u8_c1, contours, -1, 255, 3)
                    thresh = img_u8_c1

    # thresh = cv.dilate(thresh,kernel,iterations=morphologyEx_iteration)
    # thresh = cv.erode(thresh,kernel,iterations=morphologyEx_iteration)
    thresh = cv.morphologyEx(
        thresh, cv.MORPH_OPEN, kernel, iterations=morphologyEx_iteration
    )
    # if debug:
    #     imshow("thresh_erode",thresh)
    # imshow("after openting thresh_erode",thresh)

    # valid_contour: contour需要是外轮廓
    valid_contour_index = []
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.RETR_CCOMP)
    thresh_contour_len = len(contours)
    print("thresh_contour_len: ", thresh_contour_len)

    for i in range(0, thresh_contour_len):
        is_outer_contour = hierarchy[0][i][3] == -1
        if is_outer_contour:
            valid_contour_index.append(i)
            print("valid index", i)

    # 根据valid_contour创造新的marker,以处理内部contour情况
    return_masker = np.zeros(img_u8_c1.shape).astype(np.uint8)
    for j in range(0, len(valid_contour_index)):
        return_masker = get_mask_from_contour(
            return_masker, contours, valid_contour_index[j]
        )

    # 重新计算contour
    contours, hierarchy = cv2.findContours(return_masker, cv2.RETR_TREE, cv2.RETR_CCOMP)

    return return_masker, contours


# 分割路径，返回图片父路径名、图片baseName
def splitPath(img_full_path):
    img_name_tuple = os.path.split(os.path.splitext(img_full_path)[0])
    fileBaseName = img_name_tuple[1]  # test.png -> test
    parentDirName = os.path.split(img_name_tuple[0])[1]  # test_dir/test.png -> test_dir
    return parentDirName, fileBaseName


# algorithm_fun 输入: img_u8
def getU8LeftImageFromTif(img_name):
    img_16 = cv2.imread(img_name, -1)
    print("形状为: ",img_16.shape)
    img_left, img_right = cut_image(img_16, rightTrunc=100)
    img_left_u8 = u16_2_u8(img_left)

    return img_left_u8


def createDirIfNotExist(dir_abs_path):
    if not os.path.exists(dir_abs_path):
        os.mkdir(dir_abs_path)


def get_angle_from_constant_point(p0, p1, p2, debug=False):
    v1 = p0 - p1
    v2 = p2 - p1
    # print(v1)
    # print(v2)
    first_angle = cv2.fastAtan2(float(v1[1]), float(v1[0]))  # 行和列 <-> x和y
    second_angle = cv2.fastAtan2(float(v2[1]), float(v2[0]))
    # print(first_angle)
    # print(second_angle)
    result = second_angle - first_angle
    if debug:
        print(p0, p1, p2)

        print("第一个向量角度: ", first_angle)
        print("第二个向量角度: ", second_angle)

    if result < -180:
        result = result + 360
    elif result > 180:
        result = result - 2 * 180

    return result
    # return cv2.fastAtan2(float(v2[1]),float(v2[0])) - cv2.fastAtan2(float(v1[1]),float(v1[0]))

def get_distance_from_two_point(p0,p1):
    distance = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    return distance
    
# 迭代得到两点之间距离大于min_adjacent_distance的下一个点
def get_next_point_statisfing_min_distance(contour,contour_len,current_point_index,min_adjacent_distance = 3):
    current_point = contour[current_point_index][0]
    offset = 1
    next_point_index = current_point_index + offset
    if not next_point_index < contour_len:
        next_point_index = next_point_index - contour_len

    next_point = contour[next_point_index][0]
        
    distance = get_distance_from_two_point(current_point, next_point)
    while next_point_index < contour_len and distance < min_adjacent_distance:
        offset = offset + 1
        next_point_index = current_point_index + offset
        if not next_point_index < contour_len:
            break
        next_point = contour[next_point_index][0]
        distance = get_distance_from_two_point(current_point, next_point)
        # print("in loop","next_point_index: ",next_point_index,next_point,"distance: ",distance)

    if next_point_index < contour_len:
        return next_point_index
    else:
        # 无法找到满足min_adjacent_distance的点
        return -1

def get_previous_point_statisfing_min_distance(contour,contour_len,current_point_index,min_adjacent_distance = 3):
    current_point = contour[current_point_index][0]
    offset = 1
    pre_point_index = current_point_index - offset
    if not pre_point_index >= 0:
        pre_point_index = contour_len + pre_point_index
    pre_point = contour[pre_point_index][0]
    distance = get_distance_from_two_point(pre_point, current_point)
    while pre_point_index >= 0 and distance < min_adjacent_distance:
        offset = offset + 1
        pre_point_index = current_point_index - offset
        if not pre_point_index >= 0:
            break
        pre_point = contour[pre_point_index][0]
        distance = get_distance_from_two_point(pre_point,current_point)
        # print("in loop","next_point_index: ",next_point_index,next_point,"distance: ",distance)

    if pre_point_index >= 0:
        return pre_point_index
    else:
        # 无法找到满足min_adjacent_distance的点
        return -1


#cv2_circle 默认redius = 0 以绘制单个点
def cv2_circle(img_left_u8_c3,point,radius = 0,color=(255,0,0)):
    cv2.circle(img_left_u8_c3, point , radius, color)
    return img_left_u8_c3

def get_concave_point_angle(contour, current_point_index,min_adjacent_distance = 6):
    contour_len= len(contour)
    pre_point_index = get_previous_point_statisfing_min_distance(contour, contour_len, current_point_index,min_adjacent_distance)
    next_point_index = get_next_point_statisfing_min_distance(contour, contour_len, current_point_index,min_adjacent_distance)
    pre_point = contour[pre_point_index][0]
    next_point = contour[next_point_index][0]
    current_point = contour[current_point_index][0]
    angle = get_angle_from_constant_point(pre_point, current_point, next_point)
    return math.fabs(angle)


# 从contour中获得角度，以判断凹凸情况. 返回值: (index,坐标点,凹陷角度)
# min_index_distance: 判断凹点的最小距离,如果凹点Index相差小于min_index_distance,会判定为同一凹点
# 如果min_adjacent_distance 过小会导致无法抗干扰,过大也会导致一些小目标无法分割开.但是实际上,小于min_area的目标本来也不是我们的分割目标,所以我们进行忽略
def get_concave_point_from_contour(contour, img_left_u8_c1,min_adjacent_distance = 6,min_index_distance = 6,draw_help_point=False):
    img_left_u8_c3 = c1_2_c3(img_left_u8_c1)
    contour_len = len(contour)
    #list里中的每个元素为: (concave_point_index,concave_point)
    candidate_point_list = []
    
    #初始话为-1,代表没有候选凹点
    previous_candidate_index = -1

    
    contour_area = cv2.contourArea(contour, oriented=False)

    #初始设置较小的min_adjacent_distance 
    if contour_area <= 3000:
        min_adjacent_distance = 3
    elif contour_area >  3000:
        min_adjacent_distance = 3

    print("待分割图形面积: ",contour_area)
    # pre_point_index 初始设置为第一个点的index
    pre_point_index = 0
    next_point_index = get_next_point_statisfing_min_distance(contour, contour_len, pre_point_index,min_adjacent_distance)
    # print("next_point_index: ",next_point_index)

    
    for current_point_index in range(next_point_index, contour_len):

        current_point = contour[current_point_index][0]  # 中间点
        pre_point_index = get_previous_point_statisfing_min_distance(contour, contour_len, current_point_index,min_adjacent_distance)
        next_point_index = get_next_point_statisfing_min_distance(contour, contour_len, current_point_index,min_adjacent_distance)
        if pre_point_index == -1 or next_point_index == -1 :
            break
        pre_point = contour[pre_point_index][0]
        next_point = contour[next_point_index][0]
        angle = get_angle_from_constant_point(pre_point, current_point, next_point)





        # 此处凹角度应小于130
        if  angle < 0 and math.fabs(angle) < 130:
        # if  angle < 0 and math.fabs(angle) < 160:
        # if current_point_index == 20:
            # cv2_putText(img_left_u8_c3, str(current_point_index), current_point,thickness=1,fontScale=0.3)
            
            #如果这个点和之前的候选凹点距离比较远就可以取,否则忽略
            cur_pre_candidate_distance =  get_distance_from_two_point(contour[current_point_index][0], contour[previous_candidate_index][0])
            if previous_candidate_index == -1 or cur_pre_candidate_distance > min_index_distance:
            # if previous_candidate_index == -1 or (current_point_index - previous_candidate_index) > min_index_distance:
                previous_candidate_index = current_point_index
                #绘制形成凹点的辅助点
                draw_help_point = False
                if draw_help_point:
                    img_left_u8_c3 = cv2_circle(img_left_u8_c3, current_point,color=RgbColor.red)
                    img_left_u8_c3 = cv2_circle(img_left_u8_c3, pre_point,color=RgbColor.black)
                    img_left_u8_c3 = cv2_circle(img_left_u8_c3, next_point,color=RgbColor.black)
                    print("角度为: ",angle)
                    
                candidate_point_list.append((current_point_index,current_point,math.fabs(angle)))


                print(
                    "pre index:",pre_point_index,"pre 坐标: ",pre_point,"距离: ",get_distance_from_two_point(pre_point, current_point),
                    "cur index:",current_point_index,"cur 坐标: ",current_point,
                    "next index:",next_point_index,"next 坐标: ",next_point, "距离: ",get_distance_from_two_point( current_point,next_point),
                    "angle: ",angle
                    )


    if draw_help_point and len(candidate_point_list) != 0:
        imshow("ao_dian", img_left_u8_c3)
    return candidate_point_list


def cv2_putText(background_img,str_text,point,fontScale=0.5,color=(0,0,0),thickness=2):
    cv2.putText(background_img,str_text,(point[0],point[1]),cv2.FONT_HERSHEY_SIMPLEX,fontScale=fontScale,color=color,thickness=thickness)

# 将contour绘制在background_img上,
# 如果background_img为纯黑色则可以作为mask使用.默认使用纯黑色背景和常用情况兼容
# 如果contour_index = -1 则可以根据contours获得整个contours图像
def get_mask_from_contour(background_img,contours, contour_index, color=255,thickness=-1,is_use_black_background=True,background_shape=None,put_index=False):
    if is_use_black_background:
        if background_shape != None:
            background_img = np.zeros(background_shape).astype(np.uint8)
    else:
        background_img  = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)
        #如果绘制彩图,默认使用红色,边框thickness为1
        if color == 255:
            color = (255,0,0)
        if thickness == -1:
            thickness = 1

    if put_index:
        if contour_index != -1:
            x,y,w,h = cv2.boundingRect(contours[contour_index])
            cv2.putText(background_img,str(contour_index),(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,0),thickness=2)
        else:
            for i in range(0,len(contours)):
                x,y,w,h = cv2.boundingRect(contours[i])
                cv2.putText(background_img,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,0),thickness=2)





    cv2.drawContours(background_img, contours, contour_index, 255, thickness)

    return background_img


# 返回指定size大小的kernel
def get_kernel(size):
    kernel = np.ones((size, size), np.uint8)
    return kernel

def get_contour_img(img_left_u8_c1,contours,contour_index):
    # 从contour中获得roi_img
    img_left_u8_black_background = np.zeros(img_left_u8_c1.shape).astype(np.uint8)
    mask = get_mask_from_contour(img_left_u8_black_background, contours, contour_index)
    contour_img = cv2.copyTo(img_left_u8_c1, mask)
    contour_img[contour_img == 0] = 255
    return contour_img



# img: u8_c3 图像
# contours: 总共的contours
# i: contour_index
def watershed_for_contour_i(img, contours, i):
    img_u8_c3 = img
    img_left_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 从contour中获得roi_img
    img_left_u8_black_background = np.zeros(img_left_u8.shape).astype(np.uint8)
    mask = get_mask_from_contour(img_left_u8_black_background, contours, i)
    contour_img = cv2.copyTo(img_left_u8, mask)
    contour_img[contour_img == 0] = 255

    # contour_marker,marker_contours = util.get_marker_from_min_area(contour_img,morphologyEx_iteration=2)
    # util.cv2_imwrite("roi_img", contour_img)
    # 需要分水岭算法: 输入roi_img,输出,图像 contour_img ,输出 新的阈值分割的mask,输入旧的contour和img,输出,新的两个contour坐标

    thresh = constant_binary(contour_img)

    kernel_2 = get_kernel(2)
    kernel_3 = get_kernel(3)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_3, iterations=2)
    # util.imshow("opening",opening)
    sure_bg = cv.dilate(opening, kernel_3, iterations=3)
    # util.imshow("sure_bg",sure_bg)
    sure_fg, contours_of_roi = get_marker_from_min_area(contour_img)
    # util.imshow("sure_fg",sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # util.imshow("unknown", unknown)
    # 输出的marker 标记了三个类 0,1,2
    # markers中的0 总是表示背景
    # print("contours_of_roi",len(contours_of_roi))
    # connectedComponentsWithStats 函数可以获得连通域中心点坐标
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # 此时有四类
    # 0: unknown
    # 1: 背景
    # 2-x 其中的几个前景

    # print(img_left_u8.dtype)
    # print(img_left_u8.shape)
    img_left_u8_c3 = cv2.cvtColor(img_left_u8, cv2.COLOR_GRAY2BGR)
    markers = cv.watershed(img_left_u8_c3, markers)

    # 将分割线改为和背景一样的颜色黑色,将内部标签变为白色
    markers_type_len = len(contours_of_roi) + 2  # 2: 1(unknown) + 1(背景)

    markers[markers == -1] = 0
    markers[markers == 1] = 0
    for markers_type_index in range(2, markers_type_len):
        markers[markers == markers_type_index] = 255

    # util.imshow("markers",markers)
    markers = markers.astype(np.uint8)

    # 腐蚀以保证能获得正确数目的contour
    markers = cv2.erode(markers, kernel_2, iterations=1)

    # util.imshow("markers", markers)
    # util.plot_3d_img_from_matrix(markers)
    contours_markers, hierarchy_markers = cv2.findContours(
        markers, cv2.RETR_TREE, cv2.RETR_CCOMP
    )  # cv2.RETR_CCOMP 查找内外轮廓，只检测外轮廓
    contours_markers_bak = contours_markers
    # print(contours_markers)
    contour_markers_len = len(contours_markers)
    # print(contours_markers[1])
    print("contours_len:", contour_markers_len)
    mask0 = np.zeros(img_left_u8.shape).astype(np.uint8)
    mask0_img = get_mask_from_contour(mask0, contours_markers, -1)
    # imshow("mask0_img",mask0_img)
    # util.imshow("img_marked", img)
    return contours_markers_bak

    # ------------------end-----------------------------


# 获得粘连的矿石的个数
# 返回0 -- 腐蚀太过
# 返回大于1 -- 有需要分割的项目
def get_true_contour(full_img_u8_c1, contours, contour_index, kernel_size=18, iteration=1):
    contour_solid = get_mask_from_contour(None, contours, contour_index,is_use_black_background=True,background_shape=full_img_u8_c1.shape)
    # kernel_size参数对于腐蚀效果影响很大,测试18时候效果较好
    # if contour_index == 10:
    if True:
        pass

    kernel = get_kernel(kernel_size)
    # k1 = np.ones((53,53),np.uint8)
    contour_solid_erode_result = cv2.erode(contour_solid, kernel, iteration)  # 腐蚀，去除小的点
        

    erode_result_contours, erode_result_hierarchy = cv2.findContours(
        contour_solid_erode_result, cv2.RETR_TREE, cv2.RETR_CCOMP
    )
    # imshow("dst_erode",contour_solid_erode_result)
    # contour_area_erode = cv2.contourArea(erode_result_contours[0],oriented=False)
    erode_result_contours_len = len(erode_result_contours)

    # print("contours_len:",len(erode_result_contours),"area:",contour_area_erode)
    return erode_result_contours_len

#将比较近的点组队,暂时只做三个点的
def couple_closest_points(concave_points):
    concave_point_number = len(concave_points)
    if(concave_point_number <= 2):
        return -1

    distance_list = []
    distance_index_tuple = []

    for i in range(0,concave_point_number):
        for j in range(i+1,concave_point_number):
            current_distance = get_distance_from_two_point(concave_points[i][1], concave_points[j][1])
            distance_list.append(current_distance)
            distance_index_tuple.append((i,j))

    

    distance_list_np = np.array(distance_list)
    min_distance_index = np.argmin(distance_list_np)
    min_distance_index_tuple = distance_index_tuple[min_distance_index]
    print("最小距离:",min_distance_index,distance_list_np[min_distance_index],"取得最小距离的索引: ",distance_index_tuple[min_distance_index])

    return min_distance_index_tuple

def set_negative_one_for_index(matrix,index):
    len = matrix.shape[0]
    for i in range(0,len):
        matrix[i][index] = -1
        matrix[index][i] = -1
    return matrix

        
def concave_point_matching(concave_points,contour,matching_point_angle_sum_max_value=240):
    concave_points_len = len(concave_points)
    distance_matrix = np.ones((concave_points_len,concave_points_len),dtype=np.float) * -1


    #初始化距离
    for i in range(0,concave_points_len - 1):
        for j in range(i + 1,concave_points_len):
            distance = get_distance_from_two_point(concave_points[i][1], concave_points[j][1])
            distance_matrix[i][j] = distance
    
    sort_index = np.unravel_index(np.argsort(distance_matrix, axis=None), distance_matrix.shape)
    sort_index_list = [i for i in zip(sort_index[0],sort_index[1])]
    matching_result = []


    start_index = int((concave_points_len + 1) * concave_points_len / 2)
    distance_matrix_len = concave_points_len * concave_points_len


    for i in range(start_index,distance_matrix_len):
        p1_index_in_concave_points = sort_index_list[i][0]
        p2_index_in_concave_points = sort_index_list[i][1]
        p1 = concave_points[p1_index_in_concave_points][1]
        p1_angle = concave_points[p1_index_in_concave_points][2]
        p2 = concave_points[p2_index_in_concave_points][1]
        p2_angle = concave_points[p2_index_in_concave_points][2]
        p1_index_in_contour = concave_points[p1_index_in_concave_points][0]
        p2_index_in_contour = concave_points[p2_index_in_concave_points][0]

        candidate_matching_point_distance = distance_matrix[p1_index_in_concave_points][p2_index_in_concave_points]
        if candidate_matching_point_distance != -1:
            if candidate_matching_point_distance > 30:
                p1_angle = get_concave_point_angle(contour, p1_index_in_contour,min_adjacent_distance=10)
                p2_angle = get_concave_point_angle(contour, p2_index_in_contour,min_adjacent_distance=10)
            is_angle_sum_smaller_than_max_v = (p1_angle + p2_angle) < matching_point_angle_sum_max_value

            is_sibling_in_countour= is_Line_sibling_all_in_contour(*p1,*p2, contour)
            if is_use_simple or (is_sibling_in_countour and is_angle_sum_smaller_than_max_v):
            # if True:
                matching_result.append(sort_index_list[i])
                distance_matrix = set_negative_one_for_index(distance_matrix, p1_index_in_concave_points)
                distance_matrix = set_negative_one_for_index(distance_matrix, p2_index_in_concave_points)

            else:
                distance_matrix[p1_index_in_concave_points][p2_index_in_concave_points] = -1
    return matching_result
    




def distance_transform_based_watershed(img_left_u8_c1):
    # ret,thresh = cv2.threshold(img_left_u8_c1, 0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thresh = adaptiveThresholdAndErode(img_left_u8_c1)
    img_left_u8_c3 = c1_2_c3(img_left_u8_c1)
    img = img_left_u8_c3
    opening = thresh
    # util.imshow("thresh_img", thresh)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    # opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)
    
    # util.imshow("after opening", opening)
    
    
    # sure background area
    sure_bg = cv.dilate(opening, kernel,iterations=3)
    # util.imshow("sure_bg", sure_bg)
    
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    # imshow("dist_transform", dist_transform)
    ret,sure_fg = cv.threshold(dist_transform,0.3 * dist_transform.max(),255,0)
    # util.imshow("sure_fg",sure_fg)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown =  cv.subtract(sure_bg,sure_fg)
    
    # util.imshow("unknown", unknown)
    
    
    # Marker labeling
    ret,markers = cv.connectedComponents(sure_fg)
    
    print(markers.dtype)
    # print(markers)
    # util.cv2_imwrite("markers", markers)
    
    # add one to all labels so that sure background is not 0,but 1
    markers = markers + 1
    
    # Now,mark the region of unknown with zero
    
    markers[unknown == 255] = 0
    
    # util.cv2_imwrite("markers", markers)
    markers = cv.watershed(img, markers)
    print(img.shape)
    # util.cv2_imwrite("markers", markers)
    img[markers == -1] = [255,0,0]
    imshow("img_marked", img)



    

    








def get_segmentation_result_from_img_left_u8_c1(img_left_u8_c1,min_area=250):

    # print(img.shape)
    img_left_u8_c3 = cv2.cvtColor(img_left_u8_c1, cv2.COLOR_GRAY2BGR)
    img = img_left_u8_c3

    # 第一步: 自适应阈值分割
    dst = adaptiveThresholdAndErode(img_left_u8_c1)
    # dst = util.constant_binary(img_left_u8)
    # imshow("dst", dst)
    # imshow("raw_img", img_left_u8_c1)

    # 第二步: 对contour进行遍历,寻找我们的contour
    contours, hierarchy = cv2.findContours(
        dst, cv2.RETR_TREE, cv2.RETR_CCOMP
    )  # cv2.RETR_CCOMP 查找内外轮廓，只检测外轮廓
    contours_len = len(contours)
    # print(contours_len)

    dst_c3 = c1_2_c3(dst)
    # img_left_u8_c3 = dst_c3

    contours_tuple = ()

    for i in range(0, contours_len):
        # print("index",i)
        contour_area = cv2.contourArea(contours[i], oriented=False)
        mask = np.zeros(dst.shape).astype(np.uint8)
        # 是否是外轮廓
        is_outer_contour = hierarchy[0][i][3] == -1

        if is_outer_contour and contour_area > min_area:  # 是否是外轮廓，并且面积要不能低于min_area
            # 判断是否是粘连区域
            # # if i == 16:
            # if i == 20:
            # if i == 5:

            true_contours_len = get_true_contour(img_left_u8_c1, contours, i, kernel_size=18)

            contour_img =  get_contour_img(img_left_u8_c1, contours, i)
            concave_points = get_concave_point_from_contour(contours[i],contour_img)
            concave_point_numbers = len(concave_points)

            cv2.drawContours(img_left_u8_c3, contours, i, RgbColor.blue, 1)
            if concave_point_numbers > 1:
                x,y,w,h = cv2.boundingRect(contours[i])
                # cv2_putText(img_left_u8_c3, str(i), (x,y),color=RgbColor.red)
                print("轮廓Index: ",i,"---- 凹点个数: ",concave_point_numbers,"凹点: ",concave_points)
                cv2.drawContours(img_left_u8_c3, contours, i, RgbColor.red, 1)

                max_possible_stone_number = concave_point_numbers / 2 

                match_point_indexs = concave_point_matching(concave_points, contours[i])
                for match_point_index in match_point_indexs:
                    p1 = concave_points[match_point_index[0]][1]
                    p2 = concave_points[match_point_index[1]][1]
                    #绘制分割线
                    cv2.line(img_left_u8_c3,p1,p2,RgbColor.green)
                    p0_p,p1_p,p0_n,p1_n = get_line_at_distance_d(*p1,*p2, 6)
                    is_draw_helper_line = True
                    #绘制辅助线
                    if is_draw_helper_line:
                        cv2.line(img_left_u8_c3,(int(p0_p[0]),int(p0_p[1])),(int(p1_p[0]),int(p1_p[1])),RgbColor.black)
                        cv2.line(img_left_u8_c3,(int(p0_n[0]),int(p0_n[1])),(int(p1_n[0]),int(p1_n[1])),RgbColor.black)


                
                


                for concave_point in concave_points:
                    # 绘制凹点
                    # img_left_u8_c3 = cv2_circle(img_left_u8_c3, concave_point[1],color=RgbColor.blue)
                    # cv2_putText(img_left_u8_c3, str(int(concave_point[2])), concave_point[1],color=RgbColor.red,thickness=1,fontScale=0.5)
                    print("最终确定的凹点角度: ",concave_point[2])




    imshow("img_left_u8_c3_with_concave",img_left_u8_c3)
    return contours_tuple

