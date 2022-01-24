import cv2
import numpy as np
import image_processing as ip


type = 1 # 0 for test



cap = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')
#window_name = ('Face Recognition')
#cv2.namedWindow(window_name)

while True: 
    ret, frame = cap.read()
    if ret:
        key = cv2.waitKey(10)
        
        frame = ip.process_img(frame, 0)    
        frame = frame[:,:,::-1]
    if key == 27: #escape
        break

    cv2.imshow('Frame', frame)
       
    key = cv2.waitKey(20)
    #print(key)

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

