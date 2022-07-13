import cv2

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier("third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("third-party/Nose18x15.xml")

mustache = cv2.imread("mustache.png")

while True:
    ret, frame = cap.read()

    if ret==False:
        continue
    
    eyes = eye_cascade.detectMultiScale(frame, 1.3, 5)
    noses = nose_cascade.detectMultiScale(frame, 1.3, 5)

    for eye in eyes:
        x,y,w,h = eye
        glasses=cv2.resize(glasses,(w+50,h+55))
        print(glasses)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),3)
    for nose in noses:
        x,y,w,h = nose
        # mustache=cv2.resize(mustache,(w+50,h+55))
        for i in range(mustache.shape[0]):
            for j in range(mustache.shape[1]):
                if(mustache[i,j,2]>0):
                    frame[y+i,x+j,:]=mustache[i,j,:-1]
        print(mustache.shape)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),3)


    cv2.imshow("Instagram Filter", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()