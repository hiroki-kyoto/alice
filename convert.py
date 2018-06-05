# camera capture
# using opencv2 in python 2.7

import cv2 as cv
import numpy as np

def main():
	cap = cv.VideoCapture('hand.avi')
	if cap.isOpened() == False:
		return
	i = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		#cv.imshow('capture', frame)
		cv.imwrite(('hand_%d.bmp'%i), frame)
		i = i + 1
		if cv.waitKey(1) & 0xFF==ord('q'):
			break
	cap.release()
	cv.destroyAllWindows()


main()
