import numpy as np
import cv2 as cv

if __name__ == '__main__':
    cap = cv.VideoCapture('F:/dump/alcatel/video/umbrella/VID_20180412_134640.mp4')
    _, im_ = cap.read()
    cntr = 0
    sample_rate = 4
    while im_ is not None:
        im_ = cv.resize(im_, (256, 144))
        im_ = im_[:, 56:56+144]
        im_ = cv.rotate(im_, cv.ROTATE_90_CLOCKWISE)
        if cntr % sample_rate == 0:
            cv.imwrite('F:/dump/alcatel/video/umbrella/seq-out/O_' + str(cntr//sample_rate) + '.jpg', im_)
        cv.imshow('video', im_)
        cv.waitKey(10)
        _, im_ = cap.read()
        cntr += 1

