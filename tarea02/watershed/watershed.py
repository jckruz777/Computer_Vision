import numpy as np
import cv2
import pickle

# Load the initial parameters
filename = 'watershed.pkl'
infile = open(filename,'rb')
init_values = pickle.load(infile)
infile.close()

# Initial parameters
hue_labels = init_values['hue_labels']
noise_kernel_dim = init_values['noise_kernel_dim']
morph_iterations = init_values['morph_iterations']
background_iterations = init_values['background_iterations']
morph_selector = init_values['morph_selector']
thresh_factor = init_values['thresh_factor']

# Starting video streaming from /dev/video0
cap = cv2.VideoCapture(0)

while(True):
	# Capture a frame
	valid, frame = cap.read()
	if valid == True:
		# RGB-color and grayscale images from the frame
		b,g,r = cv2.split(frame)
		frame_rgb = cv2.merge([r,g,b])
		gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

		# Removing some noise with a morphologycal filter: erosion + dilation 
		kernel = np.ones((noise_kernel_dim, noise_kernel_dim), np.uint8)
		if morph_selector == 0:
			morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = morph_iterations)
		else:
			morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = morph_iterations)

		# Background area by dilation
		background = cv2.dilate(morph, kernel, iterations = background_iterations)

		# Finding foreground area with a derived representation of a binary image
		dist_transform = cv2.distanceTransform(background, cv2.DIST_L2, 3)

		# Stablishing the threshold to get a binary image out of a grayscale image
		ret, foreground = cv2.threshold(dist_transform, thresh_factor * dist_transform.max(), 255, 0)

		# Finding a region considered as unknown by two images subtraction
		foreground = np.uint8(foreground)
		unknown_reg = cv2.subtract(background, foreground)

		# Labelling the connected components on the foreground
		ret, labels = cv2.connectedComponents(foreground)

		# Add one to each label: the background is 1 (not 0)
		labels = labels + 1

		# Set the region of unknown to zero
		labels[unknown_reg == 255] = 0

		# Get the labels after the watershed algorithm
		labels = cv2.watershed(frame, labels)
		frame[labels == -1] = [255, 0, 0]
		frame[labels == 0] = [0, 0, 255]

		# Saving the segmented frame
		seg_frame = frame 

		# Map component labels to hue value
		label_hue = np.uint8(hue_labels * labels / np.max(labels))
		blank_ch = 255 * np.ones_like(label_hue)
		labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

		# Conversion to BGR for display
		labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

		# Set background label to black
		labeled_img[label_hue==0] = 0

		# Displaying the results (frame and mask)
		np_h = np.hstack((frame, labeled_img))
		np_hcat = np.concatenate((frame, labeled_img), axis=1)
		cv2.imshow('Watershed Results', np_hcat)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
