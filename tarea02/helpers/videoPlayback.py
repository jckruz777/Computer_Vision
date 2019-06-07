#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:28:15 2019

@author: luisalonsomurillorojas
"""

import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('outpy.avi')


imgCount = 0
while imgCount < 10:
    _, first_frame = cap.read()
    cv2.imwrite("test" + str(imgCount) +".jpg",first_frame)
    imgCount += 1
    # Press Q on keyboard to  exit
    if cv2.waitKey(60) & 0xFF == ord('q'):
      break
    
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(60) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()