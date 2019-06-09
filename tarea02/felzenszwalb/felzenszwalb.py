#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 00:43:40 2019

@author: luisalonsomurillorojas
"""

import numpy as np
import json
import cv2
from skimage.segmentation import felzenszwalb

cap = cv2.VideoCapture(0)

# Capture frame-by-frame
ret, frame = cap.read()

# Get serialized data
filename = 'felzenszwalb.json'
with open(filename, 'r') as f:
    data = json.load(f)
scale = data["scale"]
sigma = data["sigma"]
min_size = data["min_size"]

# Apply Felsenszwalb’s algorithm
segments_fz = felzenszwalb(frame, scale=scale, sigma=sigma, min_size=min_size)

while True:
    cv2.imshow('Mean Shift Results', segments_fz.astype(np.uint8))
    
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
