'''
Write a Python Script that captures images from your webcam video stream
Extract all faces from the image frame(using haarcascade)
Store the face information into numpy arrays

1. Read and show video stream, capture images
2. Detect faces and show bounding box
3. Flatten the largest face image(gray scale image) and save it in numpy arrays
4. Repeat the above for multiple people to generate training data
'''

import cv2
from cv2 import sort
import numpy as np

cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data = []
dataset_path = './data/'
file_name = input("Enter your name - ")
while True:
    ret, frame = cap.read()
    if ret ==False:
        continue
    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3,5)
    # print(faces)
    faces = sorted(faces,key = lambda f:f[2]*f[3], reverse=True)
    face_section=frame
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,0),3)
        #Extract region of interest means crop out the face
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        if skip%10==0:
            face_data.append(face_section)
            print(skip/10)

    cv2.imshow("Frame", frame)

    #cv2.imshow("Face frame", face_section)

    skip+=1
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy', face_data)
print("data saved at - "+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()