import util
import cv2
import cv2 as cv
import numpy as np
import cv2 
from matplotlib import pyplot as plt

coin_img_path = "/tmp/img_left_u8.png"
# coin_img_path = "/home/hhd/Downloads/coins.jpg"



def watershed_with_distance_transfrom(img_left_u8):
    img = util.c1_2_c3(img_left_u8)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray, 0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    thresh = util.adaptiveThresholdAndErode(gray)
# util.imshow("thresh_img", thresh)

# noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel,iterations=2)
    
    # util.imshow("after opening", opening)
    
    
    # sure background area
    sure_bg = cv.dilate(opening, kernel,iterations=3)
    # util.imshow("sure_bg", sure_bg)
    
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    util.imshow("dist_transform", dist_transform)
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
    util.imshow("img_marked", img)



# Add one to all labels so that sure background is not 0 ,but 1
