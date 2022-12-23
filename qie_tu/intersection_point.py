import cv2
import numpy as np
import util
import skimage
img = cv2.imread("./img/intersection_point.png",0)
print(img.shape)

def skiLine(r0,c0,r1,c1):
    rr,cc = skimage.draw.line(r0, c0, r1,c1)
    # rr = rr.astype(np.int32)
    # cc = cc.astype(np.int32)
    rr = rr.astype(np.int32)
    cc = cc.astype(np.int32)
    print("类型: ",type(rr[0]))
    return rr,cc

img = np.where(img < 100, 0, 255).astype(np.uint8)
contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])

img = util.c1_2_c3(img)
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
# 随便画一条直线
p1, p2 = (20, 200), (340, 90)
if p2[0] != p1[0]:  # 若存在斜率 y=kx+b
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - p1[0] * k
    # 求解直线和boundingbox的交点A和B
    pa, pb = (x, int(k * x + b)), ((x + w), int(k * (x + w) + b))
else:  # 若斜率不存在，垂直的直线
    pa, pb = (p1[0], y), (p1[0], y + h)
cv2.circle(img, pa, 2, (0, 255, 255), 2)
cv2.circle(img, pb, 2, (0, 255, 255), 2)
cv2.putText(img, 'A', (pa[0] - 10, pa[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
cv2.putText(img, 'B', (pb[0] + 10, pb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
for pt in zip(*skiLine(*pa, *pb)):
    # pass
    if cv2.pointPolygonTest(contours[0], ((int)(pt[0]),int(pt[1])), False) == 0:  # 若点在轮廓上
        cv2.circle(img, pt, 2, (255, 0, 0), 2)

util.imshow("img",img)
