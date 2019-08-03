import numpy as np
import cv2 as cv

if __name__ == '__main__':
    cap = cv.VideoCapture('F:/dump/alcatel/video/VID_20180407_132214.mp4')
    _, im_ = cap.read()
    cntr = 0
    sample_rate = 3
    while im_ is not None:
        im_ = cv.resize(im_, (256, 144))
        im_ = im_[:, 56:56+144]
        im_ = cv.rotate(im_, cv.ROTATE_90_CLOCKWISE)
        if cntr % sample_rate == 0:
            cv.imwrite('F:/dump/alcatel/video/seq/IM2_' + str(cntr/sample_rate) + '.jpg', im_)
        cv.imshow('video', im_)
        cv.waitKey(100)
        _, im_ = cap.read()
        cntr += 1

