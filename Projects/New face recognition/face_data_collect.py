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
import numpy as np

cap = cv2.VideoCapture(0)

#Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data =[]
dataset_path = './data/'
filename = input("Enter the name of the person: ")

while True:
    ret, frame = cap.read()
    if ret==False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key = lambda f:f[2]*f[3], reverse=True)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    skip+=1

    cv2.imshow('video', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break

face_data = np.asarray(face_data)
print("Shape of face_data2: ", face_data.shape)
face_data = face_data.reshape((face_data.shape[0], -1))
print("Shape of face_data3: ", face_data.shape)

np.save(dataset_path+filename+'.npy', face_data)
print("Data successfully saved at: ", dataset_path+filename+'.npy')

cap.release()
cv2.destroyAllWindows()