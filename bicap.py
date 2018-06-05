# camera capture
# using opencv2 in python 2.7

import cv2 as cv
import numpy as np

def main():
    cap = cv.VideoCapture(0)
    cap1 = cv.VideoCapture(1)
    if cap.isOpened() == False:
        return
    if cap1.isOpened() == False:
        return
    while True:
        ret, frame = cap.read()
        ret, frame1 = cap1.read()
        cv.imshow('capture', frame)
        cv.imshow('capture1', frame1)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cap1.release()
    cv.destroyAllWindows()

main()
