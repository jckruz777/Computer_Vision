import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

def getDescriptor(descriptor, octaves):
    if descriptor == "SIFT":
        return (cv2.xfeatures2d.SIFT_create(nOctaveLayers=octaves), "STR")
    elif descriptor == "SURF":
        return (cv2.xfeatures2d.SURF_create(nOctaves=octaves), "STR")
    elif descriptor == "ORB":
        return (cv2.ORB_create(nlevels=octaves), "BIN")
    elif descriptor == "BRISK":
        return (cv2.BRISK_create(octaves=octaves),"BIN")
    
def getKeypoints(gray1, gray2, detector):    
    
    # detect keypoints and extract local invariant descriptors from the img
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    img_ref = cv2.drawKeypoints(gray1, kps1, None)
    img_eval = cv2.drawKeypoints(gray2, kps2, None)
    
    return (img_ref, img_eval, kps1, descs1, kps2, descs2)

def getBFMatcher(ref_img, kp1, desc1, eval_img, kp2, desc2, threshold):

    # Match the features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k=2)
    
    # Apply ratio test
    good = []
    for m in matches:
        if len(m) == 2:
            if m[0].distance < threshold / 100 * m[1].distance:
                good.append(m[0])
    
    n_good_matches = len(good)
    return (good)

def getFLANNMatcher(ref_img, kp1, desc1, eval_img, kp2, desc2, threshold, alg_type):

    # Parameters for Binary Algorithms (BIN) 
    FLANN_INDEX_LSH = 6
    flann_params_bin= dict(algorithm = FLANN_INDEX_LSH,
                          table_number = 6, # 12
                          key_size = 12,     # 20
                          multi_probe_level = 1) #2

    # Parameters for String Based Algorithms (STR)
    FLANN_INDEX_KDTREE = 0
    flann_params_str = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

    search_params = dict(checks = 50)

    if alg_type == "BIN":
        flann = cv2.FlannBasedMatcher(flann_params_bin, search_params)
    else:
        flann = cv2.FlannBasedMatcher(flann_params_str, search_params)

    matches = flann.knnMatch(desc1, desc2, k = 2)

    good = []
    for m in matches:
        if len(m) == 2: 
            if m[0].distance < threshold / 100 * m[1].distance:
                good.append(m[0])

    n_good_matches = len(good)
    return (good)

def getHomography(good_matches, img1, img2, kp1, kp2):

    MIN_MATCH_COUNT = 10
    
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255,3, cv2.LINE_AA)

        return (matchesMask, img2)

    else:
        print ("Not enough matches are found: " + str(len(good_matches)) + "/" + str(MIN_MATCH_COUNT))
        matchesMask = None
        return (matchesMask, None)

def getFinalFrame(ref_img, kp1, eval_img, kp2, good_matches, matchesMask):
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    if matchesMask != None:
        img_result = cv2.drawMatches(ref_img, kp1, eval_img, kp2, good_matches, None, **draw_params)
        return img_result
    else:
        return None

parser = argparse.ArgumentParser(description='Keypoints extraction and object tracking in a still image')
parser.add_argument('--evaluation', help='Image of evaluation', default='')
parser.add_argument('--reference', help='Image of reference with the object to be detected', default="")
parser.add_argument('--descriptor', help='Descriptor algorithm: SIFT, SURF, ORB, BRISK', default='SIFT')
parser.add_argument('--matcher', help='Matcher method: Brute force (BF) or FLANN (FLANN)', default='FLANN')
parser.add_argument('--octaves', help='Number of octaves that the descriptor will use', default=4)
parser.add_argument('--mtreshold', help='Treshold for good matches.', default=70)
args = parser.parse_args()

print("Image of evaluation path = " + args.evaluation)
print("Image of reference path = " + args.reference)
print("Descriptor = " + args.descriptor)
print("Matcher = " + args.matcher)
print("Octaves = " + str(args.octaves))
print("Matcher treshold = " + str(args.mtreshold))

# Setup the video, image reference and descriptor
imgRef = cv2.imread(args.reference)
imgEval = cv2.imread(args.evaluation)
imgRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
imgEval = cv2.cvtColor(imgEval, cv2.COLOR_BGR2GRAY)
(descriptor, descriptorType) = getDescriptor(args.descriptor, args.octaves)

# Get keypoint and desciption of the frame
(img_ref, img_eval, kps1, descs1, kps2, descs2) = getKeypoints(imgRef, imgEval, descriptor)
        
# Get the match
if args.matcher == "BF":
    good = getBFMatcher(imgRef, kps1, descs1, imgEval, kps2, descs2, args.mtreshold)
elif args.matcher == "FLANN":
    good = getFLANNMatcher(imgRef, kps1, descs1, imgEval, kps2, descs2, args.mtreshold, descriptorType)
        
# Get the homography
(matchesMask, res_img) = getHomography(good, imgRef, imgEval, kps1, kps2)
        
# Get the resulting frame
result = getFinalFrame(imgRef, kps1, imgEval, kps2, good, matchesMask)

# Plot the result
graph = plt.figure()
plt.imshow(result)
graph.show()

plt.waitforbuttonpress(-1)



