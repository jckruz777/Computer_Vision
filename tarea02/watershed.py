import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame
	valid, frame = cap.read()
	if valid == True:

		b,g,r = cv2.split(frame)
		frame_rgb = cv2.merge([r,g,b])
		gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray_frame,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

		# Removing some noise with a morphologycal filter: erosion + dilation 
		kernel = np.ones((2,2), np.uint8)
		closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)

		# Background area by dilation
		background = cv2.dilate(closing, kernel, iterations=3)

		# Finding foreground area with a derived representation of a binary image
		dist_transform = cv2.distanceTransform(background, cv2.DIST_L2, 3)

		# Stablishing the threshold to get a binary image out of a grayscale image
		ret, foreground = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

		# Finding a region considered as unknown by two images subtraction
		foreground = np.uint8(foreground)
		unknown_reg = cv2.subtract(background, foreground)

		# Labelling the connected components on the foreground
		ret, labels = cv2.connectedComponents(foreground)

		# Add one to each label: the background is 1 (not 0)
		labels = labels + 1
		

		# Set the region of unknown to zero
		labels[unknown_reg == 255] = 0

		labels = cv2.watershed(frame, labels)
		frame[labels == -1] = [255, 0, 0]
		frame[labels == 0] = [0, 0, 255]

		# Diplaying the segmented frame
		cv2.imshow('frame',frame)
		
		
		# Map component labels to hue val
		label_hue = np.uint8(179 * labels / np.max(labels))
		blank_ch = 255 * np.ones_like(label_hue)
		labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

		# cvt to BGR for display
		labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

		# set bg label to black
		labeled_img[label_hue==0] = 0

		cv2.imshow('frame', labeled_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
