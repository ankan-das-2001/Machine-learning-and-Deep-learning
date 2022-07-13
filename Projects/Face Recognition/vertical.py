import cv2
import numpy as np

img = cv2.imread('photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

filter = np.array([[1,0,-1],
                  [1,0,-1],
                  [1,0,-1]])
filter2 = np.array([[1,1,1],
                  [0,0,0],
                  [-1,-1,-1]])

new_img = cv2.filter2D(src = gray,ddepth= -1, kernel = filter)  #Edge detection
new_img2 = cv2.filter2D(src = gray,ddepth= -1, kernel = filter2)    #Edge detection
cv2.imshow("New Image", new_img)
cv2.imshow("New Image2", new_img2)
cv2.waitKey(10000)
print(img.shape)
print(gray.shape)