import cv2
import numpy as np

# SETING NUMBER
PicNum = 46
getFile = 'c'
saveFile = 'cc'
ImgSize = 12

# image size cutting
# [Y_MIN, Y_MAX, X_MIN, X_MAX]
imgCut = [5, 127, 5, 262]

# alpha area size
area_min = 20
area_max = 300


src = cv2.imread("image/{}/{}.png".format(getFile,PicNum))
dst = cv2.imread("image/{}/{}.png".format(getFile,PicNum))
print(src.shape)
src = src[imgCut[0]:imgCut[1],imgCut[2]:imgCut[3]]
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),7)
ret, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in contours:
    hull = cv2.convexHull(i, clockwise=True)
    cv2.drawContours(dst, [hull], 0, (0, 0, 255), 2)

if len(contours) > 0:
    for k in range(len(contours)):
        area = cv2.contourArea(contours[k])
        print(area)

        if area > area_min and area < area_max:
            print(area)
            hull = cv2.convexHull(contours[k], clockwise=True)
            cv2.drawContours(src, [hull], 0, (0, 0, 255), 2)

            x, y, w, h = cv2.boundingRect(contours[k])

crop_img = binary[y-ImgSize:y+h+ImgSize,x-ImgSize:x+w+ImgSize]
crop_img = 255 - crop_img
dst1 = cv2.resize(crop_img, dsize=(60, 60), interpolation=cv2.INTER_AREA)

cv2.imwrite("image/{}/{}.png".format(saveFile,PicNum),dst1)

cv2.imshow("crop",crop_img)
# cv2.imshow("dst",dst)
# cv2.imshow("src",src)

# cv2.imshow("binary", binary)

cv2.imshow("dst60", dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()
