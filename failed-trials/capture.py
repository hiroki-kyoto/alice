# camera capture
# using opencv2 in python 2.7

import cv2 as cv
import numpy as np
from ctypes import *

def main():
    tools = cdll.LoadLibrary('./libtools.so')
    cap = cv.VideoCapture(0)
    if cap.isOpened() == False:
        return
    while True:
        ret, frame = cap.read()
        if not frame.flags['CONTIGUOUS']:
            frame = np.ascontiguous(frame, frame.dtype)
            print 'not contiguous!'
        frame_ptr = cast(frame.ctypes.data, POINTER(c_uint8))
        tools.change_hair_color(frame_ptr, c_int(frame.shape[0]), c_int(frame.shape[1]), c_int(frame.shape[2]))
        cv.imshow('capture', frame)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

main()
