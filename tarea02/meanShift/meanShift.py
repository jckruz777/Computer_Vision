#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Camera video
"""

import numpy as np
from sklearn.cluster import MeanShift
import json
import cv2

cap = cv2.VideoCapture(0)

#while True:
    
ret, image = cap.read()

# Get the image from the ndarray
original_shape = image.shape

# Flatten image.
X = np.reshape(image, [-1, 3])

filename = 'meanshift.json'
with open(filename, 'r') as f:
    data = json.load(f)
bandwidth = data["bandwidth"]

print("Training the mean-shift algorithm...")
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
print("Training completed!")

labels = ms.labels_
cluster_centers = ms.cluster_centers_
segmented_image = cluster_centers[np.reshape(labels, original_shape[:2])]

while True:
    cv2.imshow('Mean Shift Results', segmented_image.astype(np.uint8))
    
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#image = Image.open('bread.jpg')
#
## Image is (687 x 1025, RGB channels)
#image = np.array(image)
#original_shape = image.shape
#
## Flatten image.
#X = np.reshape(image, [-1, 3])
#
#bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
#print(bandwidth)
#
#ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ms.fit(X)
#
#labels = ms.labels_
#print(labels.shape)
#cluster_centers = ms.cluster_centers_
#print(cluster_centers.shape)
#
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)
#
#print("number of estimated clusters : %d" % n_clusters_)
#
#segmented_image = np.reshape(labels, original_shape[:2])
#
##while True:
#plt.figure(2)
#plt.subplot(1, 2, 1)
#plt.imshow(image)
#plt.axis('off')
#plt.subplot(1, 2, 2)
#plt.imshow(segmented_image)
#plt.axis('off')
#plt.show()