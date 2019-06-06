#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera video
"""

import cv2
from matplotlib import pyplot as plt

device = 0
cam = cv2.VideoCapture(0)

# Iterate until there are no more frames (artifically limiting to 10 frames)
while cam.isOpened():
    # Capture frame-by-frame
    ret, frame = cam.read()
    if ret == False:
        print("End of video")
        break;
 
    cv2.imshow('frame', frame)
    #plt.show()
 
cam.release()
cv2.destroyAllWindows()