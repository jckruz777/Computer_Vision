#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera video
"""

import pickle
import cv2

cap = cv2.VideoCapture(0)

# Capture frame-by-frame
ret, frame = cap.read()

# Our operations on the frame come here

# setup initial location of window
r,h,c,w = 90,400,279,400  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
filename = 'meanshift.pkl'
infile = open(filename,'rb')
roi_hist = pickle.load(infile)
tresh_min = pickle.load(infile)
tresh_max = pickle.load(infile)
iterations = pickle.load(infile)
pt_move = pickle.load(infile)
infile.close()

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, pt_move )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # Filtering remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dst = cv2.filter2D(dst, -1, kernel)
        _, dst = cv2.threshold(dst, tresh_min, tresh_max, cv2.THRESH_BINARY)
        mask = cv2.merge((dst, dst, dst))
        result = cv2.bitwise_and(hsv, mask)
        
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(result, (x,y), (x+w,y+h), 255,2)
        cv2.imshow("result", result)
        
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()