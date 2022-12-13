import cv2 as cv
import mediapipe as mp
import numpy as np
import time





class HandDet():
    def __init__(self,mode=False, maxhands=2, detectioncon= 0.5, trackcon= 0.5):
        # self.mode=mode
        # self.maxhands=maxhands
        # self.detectioncon=detectioncon
        # self.trackcon=trackcon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(min_detection_confidence=0.95,min_tracking_confidence=0.80)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
        
        return img


    def hand_pos(self,img,nhand=0,draw=True):

        lmlist=[]

        if self.results.multi_hand_landmarks:
            myhand= self.results.multi_hand_landmarks[nhand]
            for id,lm in enumerate(myhand.landmark):
                # print(id,hm)
                h,w,c=img.shape
                cx, cy =int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),15,(255,255,0),cv.FILLED)    
        return lmlist
    
    def finger_pos(self,lmlist):
        checklist=[4,8,12,16,20]
        status=[0,0,0,0,0]
        if lmlist:
            for ij,cl in  enumerate(checklist):
                if lmlist[cl][2]<lmlist[cl-2][2]:
                    status[ij]=1
            if lmlist[4][1]<lmlist[16][1]:
                if lmlist[4][1]>lmlist[5][1]:
                    status[0]=0
                else:
                    status[0]=1
            else:
                if lmlist[4][1]<lmlist[5][1]:
                    status[0]=0
                else:
                    status[0]=1
        return status

def main():
    pTime=0
    cTime=0
    cap=cv.VideoCapture(0)

    detect=HandDet()
    while True:
        _,img =cap.read()
        img=cv.flip(img,1)

        img = detect.findHands(img,draw=True)
        lmlist=detect.acess(img,draw=False)
        status=detect.finger_pos(lmlist)
        print(status)
        cTime= time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_DUPLEX,3,(255,0,0),3)
        cv.imshow("result",img)
        if  cv.waitKey(1) &0xFF==ord('q'):
            break


    

if __name__=="__main__":
    main()