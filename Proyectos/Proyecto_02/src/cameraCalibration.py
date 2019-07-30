from datetime import datetime
import numpy as np
import argparse
import cv2
import os
import codecs, json
import matplotlib.pyplot as plt

NO_TEST = -1
WITH_INIT_MATRIX_TEST = 0
WITH_FIX_POINT_TEST = 1
WITHOUT_INIT_MATRIX_TEST = 2
VARIABLE_TEN_CAPS_TEST = 3

def cameraCalibration(chessboardSize, captureDirectory, test_id):
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
    if test_id == WITH_FIX_POINT_TEST:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
    elif test_id == WITHOUT_INIT_MATRIX_TEST:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None, None)
    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, initCameraMatrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # Calculate error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    error_val = (mean_error/len(objpoints))
    print( "Total error: {}".format(error_val) )
    
    print('Serializaing the calibration results...')
    results =	{
        "mtx": mtx.tolist(),
        "dist": dist.tolist()
    }
    json.dump(results, codecs.open('../matrix_params/calibration-' + str(datetime.timestamp(datetime.now())) + '.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    # Return the error value
    return error_val

def main():
    # Parameters
    CHESSBOARD_SIZE = (7, 7)
    CAPTURE_DIRECTORY = './'
    
    # Setup the command line arguments
    parser = argparse.ArgumentParser(description='Camera calibration process')
    parser.add_argument('--chessboardSize', help='Size of the chessboard (Default (7, 7))', nargs='+', type=int)
    parser.add_argument('--captureDirectory', help='Path to the directory to store the captures (Default is ./)', default='./')
    parser.add_argument('--testid', help='ID of the test to be performed', default='-1')
    args = parser.parse_args()

    CAPTURE_DIRECTORY = str(args.captureDirectory)
    TEST_ID = int(args.testid)

    if args.chessboardSize != None:
        CHESSBOARD_SIZE = tuple(args.chessboardSize)

    if TEST_ID == NO_TEST:
        cameraCalibration(CHESSBOARD_SIZE, CAPTURE_DIRECTORY, 0)

    elif TEST_ID == VARIABLE_TEN_CAPS_TEST:
        timestamp_cap = str(datetime.timestamp(datetime.now()))
        res_caps_file_name = "../results/results_iterations_file_" + timestamp_cap + ".txt"
        results_file = open(res_caps_file_name, "w")
        print("----------------------------------------------")
        print("Starting -variable number of captures- test...")
        print("----------------------------------------------")
        for test_caps_i in range(10):
            CAPTURE_DIRECTORY = '../test_dirs/test_dir' + str(test_caps_i + 1) + '/'
            # Start the calibration process
            print("Calculating the error value for" + str(test_caps_i + 1) + "captures...")
            error_value = cameraCalibration(CHESSBOARD_SIZE, CAPTURE_DIRECTORY, 0)
            results_file.write(str(test_caps_i + 1) + ":" + str(error_value) + '\n')
        results_file.close()
        # Graph information
        x, y = np.loadtxt(res_caps_file_name, delimiter=':', unpack=True)
        plt.plot(x,y, label='Error Behavior - Test' + str(args.testid))
        plt.xlabel('Number of captured images')
        plt.ylabel('Calibration Error')
        plt.legend()
        plt.savefig('../results/num_captures_error_' + timestamp_cap + '.png')
        plt.show()
    else:
        timestamp_it = str(datetime.timestamp(datetime.now()))
        res_it_file_name = "../results/results_iterations_file_" + timestamp_it + ".txt"
        results_file = open(res_it_file_name, "w")
        print("------------------------------------------")
        print("Starting -variable 9_capture set- test...")
        print("------------------------------------------")
        for test_index in range(10):
            CAPTURE_DIRECTORY = '../test_dirs/test_set' + str(test_index + 1) + '/'
            # Start the calibration process
            print("Calculating the error value for set #" + str(test_index + 1) + "...")
            error_value = cameraCalibration(CHESSBOARD_SIZE, CAPTURE_DIRECTORY, TEST_ID)
            results_file.write(str(test_index + 1) + ":" + str(error_value) + '\n')
        results_file.close()
        # Graph information
        x, y = np.loadtxt(res_it_file_name, delimiter=':', unpack=True)
        plt.plot(x,y, label='Error Behavior - Test' + str(args.testid))
        plt.xlabel('9-Images set ID')
        plt.ylabel('Calibration Error')
        plt.legend()
        plt.savefig('../results/set_it_error_' + timestamp_it + '.png')
        plt.show()

if __name__ == "__main__":
    main()
