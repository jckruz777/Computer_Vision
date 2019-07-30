import numpy as np
import argparse
import cv2
import json

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def streamTest(device, calibrationPath, chessboardSize):
    # Load the calibration parameters
    with open(calibrationPath, 'r') as file:
        data = json.load(file)
    mtx = np.asarray(data['mtx'])
    dist = np.asarray(data['dist'])
    
    # Stop criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Get the camera device
    cap = cv2.VideoCapture(device)

    while(True):
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
            objp = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:chessboardSize[1],0:chessboardSize[0]].T.reshape(-1,2)
            axis = np.float32([[4,0,0], [0,4,0], [0,0,-4]]).reshape(-1,3)
            
            ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
            if ret == True:
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                frame = draw(frame,corners,imgpts)
                

            # Display the resulting frame
            cv2.imshow('Calibration test',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            print('Cannnot get more frame from the camera device')
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parameters
    DEVICE = 0
    CALIBRATE_FILENAME = 'chessboard'
    CHESSBOARD_SIZE = (7, 7)
    
    # Setup the command line arguments
    parser = argparse.ArgumentParser(description='Test script for the camera calibration')
    parser.add_argument('--device', help='Camera device id (Default is 0)', default=0, type=int)
    parser.add_argument('--calibrateFilename', help='Calibration config filename', default='')
    parser.add_argument('--chessboardSize', help='Size of the chessboard (Default (7, 7))', nargs='+', type=int)
    args = parser.parse_args()
    
    DEVICE = int(args.device)
    CALIBRATE_FILENAME = str(args.calibrateFilename)
    if args.chessboardSize != None:
        CHESSBOARD_SIZE = tuple(args.chessboardSize)
    
    # Test the calibration
    streamTest(DEVICE, CALIBRATE_FILENAME, CHESSBOARD_SIZE)

if __name__ == "__main__":
    main()
