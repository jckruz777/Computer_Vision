#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:43:53 2019

@author: luisalonsomurillorojas
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera video
"""

import cv2
import pickle

frame = cv2.imread("../meanShift/test_frames/test7.jpg")

#meanshift parameters
iterations = 10
pt_move = 1

# treshold parameters
tresh_min = 220
tresh_max = 255
    
# setup initial location of window
r,h,c,w = 182,505,279,547  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#NOTE: uncomment this code just for test
#while True:
#    cv2.imshow("ROI",hsv_roi)
#    
#    # Press Q on keyboard to  exit
#    if cv2.waitKey(60) & 0xFF == ord('q'):
#      break
#
## When everything done, release the capture
#cv2.destroyAllWindows()

# Serializing
filename = 'camshift.pkl'
outfile = open(filename,'wb')
pickle.dump(roi_hist,outfile)
pickle.dump(tresh_min,outfile)
pickle.dump(tresh_max,outfile)
pickle.dump(iterations,outfile)
pickle.dump(pt_move,outfile)
outfile.close()