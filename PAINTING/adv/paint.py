import cv2 as cv
import os
import mediapipe as mp
import numpy as np
import handtrackingmodule as htm

photopath="sources"
myfiles=os.listdir(photopath)
photolist=[]
for impath in myfiles:
    image=cv.imread(f'{photopath}/{impath}')
    photolist.append(image)
header=photolist[0]


wcap, hcap=1280,800
cap=cv.VideoCapture(0)
cap.set(3,wcap)
cap.set(4,hcap)


imgcanvas=np.zeros((720,1280,3), dtype='uint8')

detect=htm.HandDet()
pos=[]
colors=[(0,0,255),(0,255,0),(255,37,3),(0,255,255),(191,64,191),(226,181,0),(0,165,255),(52,71,21),(0,0,0)]
paint=colors[0]
xp=0
yp=0
size=10


while True:
    _,img=cap.read()
    img=cv.flip(img,1)
    img=detect.findHands(img,draw=False)
    lmlist=detect.hand_pos(img, draw=False)
    if lmlist:
        x1,y1 =lmlist[8][1:]
        x2,y2 =lmlist[12][1:]

    fingers=detect.finger_pos(lmlist)
    if fingers[1] and fingers[2]:
        xp, yp=0, 0
        cv.rectangle(img,(x1,y1-25),(x2,y2+25),(255,0,255),cv.FILLED)
        if x1<210:
            if 0<=x1<=105:
                if 0<=y1<=110:
                    header=photolist[0]
                    paint=colors[0]
                    size=10
                if 110<=y1<=212:
                    header=photolist[2]
                    paint=colors[2]
                    size=10
                if 212<=y1<=315:
                    header=photolist[4]
                    paint=colors[4]
                    size=10
                if 315<=y1<=400:
                    header=photolist[6]
                    paint=colors[6]
                    size=10
                if 400<=y1<=520:
                    header=photolist[8]
                    paint=colors[8]
                    size=100
            if 105<=x1<=210:
                if 0<=y1<=110:
                    header=photolist[1]
                    paint=colors[1]
                    size=10
                if 110<=y1<=212:
                    header=photolist[3]
                    paint=colors[3]
                    size=10
                if 212<=y1<=315:
                    header=photolist[5]
                    paint=colors[5]
                    size=10
                if 315<=y1<=400:
                    header=photolist[7]
                    paint=colors[7]
                    size=10
                if 400<=y1<=520:
                    header=photolist[8]
                    paint=colors[8]
                    size=100

    if fingers[1] and fingers[2]==0:
        if xp==0 and yp==0:
            xp=x1
            yp=y1
        if x1>230:
            cv.circle(img,(x1,y1),15,paint,cv.FILLED)
            cv.line(img,(xp,yp),(x1,y1),(0,0,225),size)
            cv.line(imgcanvas,(xp,yp),(x1,y1),paint,size)
        xp, yp= x1, y1

    img[:580,0:210]= header
    imggray= cv.cvtColor(imgcanvas,cv.COLOR_BGR2GRAY)
    _, imginv= cv.threshold(imggray,50,255, cv.THRESH_BINARY_INV)
    imginv=cv.cvtColor(imginv,cv.COLOR_GRAY2BGR)
    img=cv.bitwise_and(img,imginv)
    img=cv.bitwise_or(img,imgcanvas)
    cv.imshow("paint",img)
    if cv.waitKey(1) & 0xFF== ord("q"):
        break
