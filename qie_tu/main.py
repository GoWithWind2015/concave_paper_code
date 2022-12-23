#coding=utf-8
import os.path

import numpy as np
import  util
import  matplotlib.pyplot as plt
import pylab
import cv2
import cv2 as cv
import timeit
import sys
import time
from typing import Tuple
from util import  global_variable
import watershed

start = time.process_time()
workspace_base_dir  = "/home/hhd/workspace/longi_coal_picture/"

img_name = "./img/2307_01652_33_032.tif"
img_name = "./img/2316_00239_26_027.tif"
img_name = "./img/2322_00241_21_035.tif"
img_name = "./img/2357_00254_20_034.tif"
img_name = "./img/2815_01984_20_018.tif"
img_name = "./img/2820_00291_18_016.tif"
img_name = "./img/5134_00707_18_005.tif"
img_name = "./img/5819_00841_18_002.tif"

global_variable = {}


print("命令行参数: " + sys.argv[1])


if(sys.argv[1] == "simple"):
    util.is_use_simple = True
if(sys.argv[1] == "watershed"):
    util.is_use_watershed = True




def get_segmetation_result(img_name):
    print("===================新的图像========================================")
    print(img_name)
    img_left_u8 = util.getU8LeftImageFromTif(img_name)
    if util.is_use_watershed:
        watershed.watershed_with_distance_transfrom(img_left_u8)
    else:
        segmentation_contours_tuple = util.get_segmentation_result_from_img_left_u8_c1(img_left_u8)

get_segmetation_result(img_name)


end = time.process_time()

print("程序运行时间: " ,end - start ," seconds" )
