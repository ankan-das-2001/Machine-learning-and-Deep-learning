'''
Representing an image file
'''
import numpy as np
import cv2
#Representing a pixle
img = np.zeros((300,300), dtype=np.uint8)
cv2.imshow("Black_image.png",img)
cv2.waitKey()
cv2.destroyAllWindows()