import cv2
import mediapipe as mp 

hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
handlmsstyle = draw.DrawingSpec(color = (0,0,255),thickness = 5)
handconstyle = draw.DrawingSpec(color = (0,255,0),thickness = 5)
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture("rtsp://admin:xirui360@192.168.3.136:554/Streaming/Channels/101")
while True:
    ret,img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                draw.draw_landmarks(img,handlms,mp.solutions.hands.HAND_CONNECTIONS,handlmsstyle,handconstyle)
        cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q') :
        break