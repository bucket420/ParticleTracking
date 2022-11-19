import cv2 as cv
import time

video = cv.VideoCapture('./Examples/20200730/025Hz-01.avi')
FPS = video.get(cv.CAP_PROP_FPS)
while True:
    hasNextFrame, frame = video.read()
    
    if hasNextFrame:
        cv.imshow('Example', frame)
        
    if cv.waitKey(20) & 0xFF==ord(' '):
        break

    
video.release()
cv.destroyAllWindows()
