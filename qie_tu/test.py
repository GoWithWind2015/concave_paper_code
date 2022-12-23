import time

import cv2
import cv2 as cv
import numpy as np
import util
from matplotlib import pyplot as plt
from PIL import Image

# coin_img_path = "/home/hhd/Downloads/roi_img.png"


start = time.process_time()

# coin_img_path = "/tmp/img_left_u8.png"
# img = cv2.imread(coin_img_path)

# util.get_segmentation_result_from_img_left_u8_c3(img)

        


# 二维数组
x = np.array([[0, 3], [2, 2]])
# n维数组元素排序后的索引
# print(np.argsort(x, axis=None))
ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
print('多维数组拉伸为一维后排序的索引为：{}'.format(ind))
print('将多维数组拉伸为一维后进行排序：{}'.format(x[ind]))

for i in range(0,3):
    print(i)


end = time.process_time()

print("程序运行时间: ", end - start, " seconds")
