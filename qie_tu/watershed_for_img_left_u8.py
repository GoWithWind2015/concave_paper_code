import util
import cv2
import cv2 as cv
import numpy as np
import cv2 
from matplotlib import pyplot as plt

# coin_img_path = "/home/hhd/Downloads/coins.jpg"

# coin_img_path = "/tmp/img_left_u8.png"
coin_img_path = "/home/hhd/Downloads/roi_img.png"
img = cv2.imread(coin_img_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# ret,thresh = cv2.threshold(gray, 0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
thresh = util.adaptiveThresholdAndErode(gray)
# thresh = util.constant_binary(gray)

# util.imshow("thresh_img", thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)

util.imshow("after opening", opening)



# sure background area
sure_bg = cv.dilate(opening, kernel,iterations=3)
# util.imshow("sure_bg", sure_bg)


# Finding sure foreground area
#------------------ 距离变换开始 -------------------
# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# util.imshow("dist_transform", dist_transform)
# ret,sure_fg = cv.threshold(dist_transform,0.25 * dist_transform.max(),255,0)

#---------------- 距离变换结束 ---------------
#-------------- 腐蚀开始-----------------
# 腐蚀和距离变化后threshold配分水岭算法都能够很好的进行距离的变换,难的是符合和距离变换后的参数怎么选?
# sure_fg = cv.erode(thresh,kernel,iterations=5)

#---------------腐蚀结束----------------

# util.imshow("sure_fg",sure_fg)

# Finding unknown region

# sure_fg = np.uint8(sure_fg)

# 使用基于最小面积的方法来求sure_fg

sure_fg,contours = util.get_marker_from_min_area(gray)
unknown =  cv.subtract(sure_bg,sure_fg)
util.imshow("sure_fg using get_marker_from_min_area", sure_fg)

util.imshow("unknown", unknown)


# Marker labeling
ret,markers = cv.connectedComponents(sure_fg)
# print(markers)
# util.cv2_imwrite("markers", markers)

# add one to all labels so that sure background is not 0,but 1
markers = markers + 1

# Now,mark the region of unknown with zero

markers[unknown == 255] = 0

# util.cv2_imwrite("markers", markers)
markers = cv.watershed(img, markers)
# util.cv2_imwrite("markers", markers)
img[markers == -1] = [255,0,0]
util.imshow("img_marked", img)
util.cv2_imwrite("img_marked", img)




# Add one to all labels so that sure background is not 0 ,but 1
