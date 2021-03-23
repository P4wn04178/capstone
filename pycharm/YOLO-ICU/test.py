import cv2
import numpy as np

video_src = "../1744.mp4"
cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임
delay = int(1000/fps) # 계산 딜레이

cascade_front = cv2.CascadeClassifier("haarcascade_frontface.xml")
# cascade_front = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")
cascade_profile = cv2.CascadeClassifier("haarcascade_profileface.xml")

tracker = cv2.TrackerCSRT_create()
tracking_State = 0
tracking_ROI = (0,0,0,0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("can't read video")
        break

    if tracking_State == 0:
        image = frame.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade_front.detectMultiScale(gray, 1.3, 5)
        profile_faces = cascade_profile.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x,y,w,h = faces[0]
            tracking_ROI = (x,y,w,h)
            # cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255),2,1)
            tracking_State = 1
            print(faces)

    elif tracking_State == 1:
        inits = tracker.init(frame, tracking_ROI)
        if inits == None:
            tracking_State = 2
            print("tracking..")
        else:
            tracking_State = 0
            print("tracking failed")

    elif tracking_State == 2:
        ok, tracking_ROI = tracker.update(frame)
        if ok:
            p1 = (int(tracking_ROI[0]), int(tracking_ROI[1]))
            p2 = (int(tracking_ROI[0] + tracking_ROI[1]), int(tracking_ROI[1] + tracking_ROI[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0),2,1)
        else:
            print("failed")
            tracking_State = 0

    cv2.imshow("main", frame)
    if cv2.waitKey(delay) & 0xff == ord('q'):
        break

else:
    print("can't find file")
cap.release()