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
WITH_FIX_ASPECT_RATIO_TEST = 3
VARIABLE_TEN_CAPS_TEST = 4

test_dict = [ "With Initialization Matrix",                    #Test 0
              "With Fix Point Flag",                           #Test 1
              "Without Initialization Matrix",                 #Test 2
              "With Fix Aspect Ratio Flag",                    #Test 3
              "Increasing the Number of Captured Images" ]     #Test 4

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
    if test_id == WITH_FIX_POINT_TEST: # Test_ID=1
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
    elif test_id == WITHOUT_INIT_MATRIX_TEST: # Test_ID=2
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None, None)
    elif test_id == WITH_FIX_ASPECT_RATIO_TEST: # Test_ID=3
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)
    else: # Test_ID=0
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, initCameraMatrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    num_points = len(objpoints)

    # Calculate error
    acc_error = 0
    highest_error = 0
    error_array = np.zeros(num_points)
    for i in range(num_points):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        if error > highest_error:
            highest_error = error
        acc_error += error
        error_array[i] = error
    mean_error = (acc_error/num_points)
    desvesta = np.std(error_array)
    median = np.median(error_array)


    print( "Mean error: {}".format(mean_error) )
    
    print('Serializaing the calibration results...')
    results =	{
        "mtx": mtx.tolist(),
        "dist": dist.tolist()
    }
    json.dump(results, codecs.open('../matrix_params/calibration-' + str(datetime.timestamp(datetime.now())) + '.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    # Return the error stats values    
    return (mean_error, median, desvesta, highest_error)


def plot_error_stat (test_name, filename, x_label, y_label, plot_label, timestamp_cap, test_ID):
    # Graph information
    x, y = np.loadtxt(filename, delimiter=':', unpack=True)
    plt.plot(x,y, label=plot_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig('../results/test_' + str(test_ID) + '_' + test_name + '_' + timestamp_cap + '.png')
    plt.show()

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
        res_caps_mean_fname = "../results/results_mean_file_" + timestamp_cap + ".txt"
        res_caps_median_fname = "../results/results_median_file_" + timestamp_cap + ".txt"
        res_caps_desv_fname = "../results/results_desv_file_" + timestamp_cap + ".txt"
        res_caps_herror_fname = "../results/results_herror_file_" + timestamp_cap + ".txt"
        results_mean_file = open(res_caps_mean_fname, "w")
        results_median_file = open(res_caps_median_fname, "w")
        results_desv_file = open(res_caps_desv_fname, "w")
        results_herror_file = open(res_caps_herror_fname, "w")

        print("----------------------------------------------")
        print("Starting -variable number of captures- test...")
        print("----------------------------------------------")
        for test_caps_i in range(3, 10):
            CAPTURE_DIRECTORY = '../test_dirs/test_dir' + str(test_caps_i + 1) + '/'
            # Start the calibration process
            print("Calculating the error value for" + str(test_caps_i + 1) + "captures...")
            mean_error, median, desvesta, highest_error = cameraCalibration(CHESSBOARD_SIZE, CAPTURE_DIRECTORY, 0)
            results_mean_file.write(str(test_caps_i + 1) + ":" + str(mean_error) + '\n')
            results_median_file.write(str(test_caps_i + 1) + ":" + str(median) + '\n')
            results_desv_file.write(str(test_caps_i + 1) + ":" + str(desvesta) + '\n')
            results_herror_file.write(str(test_caps_i + 1) + ":" + str(highest_error) + '\n')
        results_mean_file.close()
        results_median_file.close()
        results_desv_file.close()
        results_herror_file.close()
        # Graph information
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Test ' + str(TEST_ID) + ': ' + test_dict[TEST_ID])
        x, y = np.loadtxt(res_caps_mean_fname, delimiter=':', unpack=True)
        axs[0, 0].plot(x,y, label='Mean Behavior - Test_' + str(args.testid))
        axs[0, 0].set_title("Mean Error")
        axs[0, 0].set(xlabel='Number of captured images', ylabel='Calibration Mean Error')
        x, y = np.loadtxt(res_caps_median_fname, delimiter=':', unpack=True)
        axs[0, 1].plot(x,y, label='Median Behavior - Test_' + str(args.testid))
        axs[0, 1].set_title("Median Error")
        axs[0, 1].set(xlabel='Number of captured images', ylabel='Calibration Median Error')
        x, y = np.loadtxt(res_caps_desv_fname, delimiter=':', unpack=True)
        axs[1, 0].plot(x,y, label='Stand_Dev Behavior - Test_' + str(args.testid))
        axs[1, 0].set_title("Standard Dev. Error")
        axs[1, 0].set(xlabel='Number of captured images', ylabel='Calibration Standard Dev. Error')
        x, y = np.loadtxt(res_caps_herror_fname, delimiter=':', unpack=True)
        axs[1, 1].plot(x,y, label='HError Behavior - Test_' + str(args.testid))
        axs[1, 1].set_title("Highest Error Value")
        axs[1, 1].set(xlabel='Number of captured images', ylabel='Calibration Highest Error Value')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../results/test_' + str(TEST_ID) + '_' + timestamp_cap + '.svg')
        plt.show()
        """
        plot_error_stat ('mean', res_caps_mean_fname, 'Number of captured images', 'Calibration Mean Error', 
                        'Mean Behavior - Test_' + str(args.testid), timestamp_cap, TEST_ID)
        plot_error_stat ('median', res_caps_median_fname, 'Number of captured images', 'Calibration Median Error', 
                        'Median Behavior - Test_' + str(args.testid), timestamp_cap, TEST_ID)
        plot_error_stat ('desv', res_caps_desv_fname, 'Number of captured images', 'Calibration Standard_Dev. Error', 
                        'Desv Behavior - Test_' + str(args.testid), timestamp_cap, TEST_ID)
        plot_error_stat ('herror', res_caps_herror_fname, 'Number of captured images', 'Calibration Highest Error Value', 
                        'HError Behavior - Test_' + str(args.testid), timestamp_cap, TEST_ID)
        """
    else:
        timestamp_it = str(datetime.timestamp(datetime.now()))
        res_it_mean_file_name = "../results/results_it_mean_file_" + timestamp_it + ".txt"
        res_it_median_file_name = "../results/results_it_median_file_" + timestamp_it + ".txt"
        res_it_desv_file_name = "../results/results_it_desv_file_" + timestamp_it + ".txt"
        res_it_herror_file_name = "../results/results_it_herror_file_" + timestamp_it + ".txt"
        results_mean_file = open(res_it_mean_file_name, "w")
        results_median_file = open(res_it_median_file_name, "w")
        results_desv_file = open(res_it_desv_file_name, "w")
        results_herror_file = open(res_it_herror_file_name, "w")
        print("------------------------------------------")
        print("Starting -variable 9_capture set- test...")
        print("------------------------------------------")
        for test_index in range(10):
            CAPTURE_DIRECTORY = '../test_dirs/test_set' + str(test_index + 1) + '/'
            # Start the calibration process
            print("Calculating the error value for set #" + str(test_index + 1) + "...")
            mean_error, median, desvesta, highest_error = cameraCalibration(CHESSBOARD_SIZE, CAPTURE_DIRECTORY, TEST_ID)
            results_mean_file.write(str(test_index + 1) + ":" + str(mean_error) + '\n')
            results_median_file.write(str(test_index + 1) + ":" + str(median) + '\n')
            results_desv_file.write(str(test_index + 1) + ":" + str(desvesta) + '\n')
            results_herror_file.write(str(test_index + 1) + ":" + str(highest_error) + '\n')
        results_mean_file.close()
        results_median_file.close()
        results_desv_file.close()
        results_herror_file.close()
        # Graph information
        fig, axs = plt.subplots(2, 2)
        fig.suptitle('Test ' + str(TEST_ID) + ': ' + test_dict[TEST_ID])
        x, y = np.loadtxt(res_it_mean_file_name, delimiter=':', unpack=True)
        axs[0, 0].plot(x,y, label='Mean Behavior - Test_' + str(args.testid))
        axs[0, 0].set_title("Mean Error")
        axs[0, 0].set(xlabel='9-Images set ID', ylabel='Calibration Mean Error')
        x, y = np.loadtxt(res_it_median_file_name, delimiter=':', unpack=True)
        axs[0, 1].plot(x,y, label='Median Behavior - Test_' + str(args.testid))
        axs[0, 1].set_title("Median Error")
        axs[0, 1].set(xlabel='9-Images set ID', ylabel='Calibration Median Error')
        x, y = np.loadtxt(res_it_desv_file_name, delimiter=':', unpack=True)
        axs[1, 0].plot(x,y, label='Stand_Dev Behavior - Test_' + str(args.testid))
        axs[1, 0].set_title("Standard Dev. Error")
        axs[1, 0].set(xlabel='9-Images set ID', ylabel='Calibration Standard Dev. Error')
        x, y = np.loadtxt(res_it_herror_file_name, delimiter=':', unpack=True)
        axs[1, 1].plot(x,y, label='HError Behavior - Test_' + str(args.testid))
        axs[1, 1].set_title("Highest Error Value")
        axs[1, 1].set(xlabel='9-Images set ID', ylabel='Calibration Highest Error Value')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('../results/test_' + str(TEST_ID) + '_' + timestamp_it + '.svg')
        plt.show()
        """
        plot_error_stat ('mean', res_it_mean_file_name, '9-Images set ID', 'Calibration Mean Error', 
                        'Mean Behavior - Test_' + str(args.testid), timestamp_it, TEST_ID)
        plot_error_stat ('median', res_it_median_file_name, '9-Images set ID', 'Calibration Median Error', 
                        'Median Behavior - Test_' + str(args.testid), timestamp_it, TEST_ID)
        plot_error_stat ('desv', res_it_desv_file_name, '9-Images set ID', 'Calibration Standard_Dev. Error', 
                        'Desv Behavior - Test_' + str(args.testid), timestamp_it, TEST_ID)
        plot_error_stat ('herror', res_it_herror_file_name, '9-Images set ID', 'Calibration Highest Error Value', 
                        'HError Behavior - Test_' + str(args.testid), timestamp_it, TEST_ID)
        """

if __name__ == "__main__":
    main()
