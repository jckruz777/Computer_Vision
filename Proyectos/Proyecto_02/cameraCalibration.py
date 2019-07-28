from datetime import datetime
import numpy as np
import argparse
import cv2
import os
import codecs, json

def cameraCalibration(chessboardSize, captureDirectory):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[1],0:chessboardSize[0]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Get the list of images
    imgList = os.listdir(captureDirectory)
    gray = None
    
    print('Loading the object point and image points...')
    # Iterate over the images
    for imgFile in imgList:
        # Load the image
        img = cv2.imread(captureDirectory + imgFile)
        
        if img.size > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            objpoints.append(objp)
            
            # Get the correspondencies
            ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
            
            if ret:
                # Optimize the corner points or correspondencies
                imgpoints.append(corners)
        else:
            print('Error loading the file')
            
    print('Initilizing the camera matrix...')
    # Init the camera parameters
    initCameraMatrix = cv2.initCameraMatrix2D(objpoints, imgpoints, gray.shape)
    
    print('Performing the camera calibration...')
    # Perform the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, initCameraMatrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    
    # Calculate error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Total error: {}".format(mean_error/len(objpoints)) )
    
    print('Serializaing the calibration results...')
    results =	{
        "mtx": mtx.tolist(),
        "dist": dist.tolist()
    }
    json.dump(results, codecs.open('calibration-' + str(datetime.timestamp(datetime.now())) + '.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

def main():
    # Parameters
    CHESSBOARD_SIZE = (7, 6)
    CAPTURE_DIRECTORY = './'
    
    # Setup the command line arguments
    parser = argparse.ArgumentParser(description='Camera calibration process')
    parser.add_argument('--chessboardSize', help='Size of the chessboard (Default (7, 6))', nargs='+', type=int)
    parser.add_argument('--captureDirectory', help='Path to the directory to store the captures (Default is ./)', default='./')
    args = parser.parse_args()

    CAPTURE_DIRECTORY = str(args.captureDirectory)
    if args.chessboardSize != None:
        CHESSBOARD_SIZE = tuple(args.chessboardSize)
    
    # Start the calibration process
    cameraCalibration(CHESSBOARD_SIZE, CAPTURE_DIRECTORY)

if __name__ == "__main__":
    main()
