#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:43:40 2019

@author: luisalonsomurillorojas
"""

import numpy as np
import pickle
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    
    # setup initial location of window
    r,h,c,w = 90,400,279,400  # simply hardcoded the values
    track_window = (c,r,w,h)

    # set up the ROI for tracking
    filename = 'meanShift'
    infile = open(filename,'rb')
    roi_hist = pickle.load(infile)
    infile.close()
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    
    while(1):
        ret ,frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            
            # Filtering remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dst = cv2.filter2D(dst, -1, kernel)
            _, dst = cv2.threshold(dst, 250, 255, cv2.THRESH_BINARY)
            
            mask = cv2.merge((dst, dst, dst))
            result = cv2.bitwise_and(hsv, mask)
            
            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(result,[pts],True, 255,2)
            
            #cv2.imshow("Mask", dst)
            cv2.imshow('Result',img2)
            
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)
                cv2.imwrite(chr(k)+"2.jpg",dst)
        else:
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()