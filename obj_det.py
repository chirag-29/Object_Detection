import cv2

import numpy as np
print("hello")

cap = cv2.VideoCapture(0)
template = cv2.imread('template.jpg',0)
#cv2.imshow('image',template)

def ORBdetector(image,template):
    
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create() 
    (kp1,des1) = orb.detectAndCompute(image1,None)
    (kp2,des2) = orb.detectAndCompute(template,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
    
    matches = bf.match(des1,des2)

    matches = sorted(matches,key=lambda val: val.distance)

    return len(matches)

count = 0

st = "Count = "+str(count)
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    top_leftx = int(width / 3)
    top_lefty = int((height / 2) + (height / 4))
    bottom_rightx = int((width / 3) * 2)
    bottom_righty = int((height / 4))
    
    cv2.rectangle(frame, (top_leftx,top_lefty), (bottom_rightx,bottom_righty), (255,0,0), 3)
    
    cropped = frame[bottom_righty:top_lefty , top_leftx:bottom_rightx]

    matches = ORBdetector(cropped,template)
    
    
    string = "Matches = "+str(matches)
    cv2.putText(frame,string,(50,600),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,150),2)
    
    cv2.putText(frame,st,(50,690),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,150),2)
    threshold = 310
    
    if matches > threshold:
        count += 1
        st = "Count = "+str(count) 
        cv2.rectangle(frame,(top_leftx,top_lefty),(bottom_rightx,bottom_righty),(0,255,0),5)
        cv2.putText(frame,'object found',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
        cv2.putText(frame,st,(50,690),cv2.FONT_HERSHEY_COMPLEX,2,(250,0,150),2)

    cv2.imshow("object detection using ORB",frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()








