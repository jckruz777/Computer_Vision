import numpy as np
import argparse
import time
import cv2

def captureChessboard(device, captures, interval, chessboardSize, winSize, showChessboardCorner, captureDirectory, captureFilename):
    # Get the camera device
    cap = cv2.VideoCapture(device)

    # Criteria for corner points optimization process
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialization
    numCaptures = 0
    timeCounter = 0
    start = time.time()

    while(numCaptures < captures):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = gray
        
            # Get the correspondencies
            ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
            
            if ret:
                # Optimize the corner points or correspondencies
                corners2 = cv2.cornerSubPix(gray, corners, (winSize[0] * 2 + 1, winSize[1] * 2 + 1), (-1,-1), criteria)
                
                if timeCounter >= interval:
                    print('Taking new capture')
                    
                    # Save image
                    imagePath = captureDirectory + captureFilename + '_' + str(numCaptures) + '.png'
                    status = cv2.imwrite(imagePath, gray)
                
                    # If success, increment the capture count
                    if status:
                        numCaptures = numCaptures + 1
                        start = time.time()
                    else:
                        print('Error while storing a new capture.')
                
                if showChessboardCorner:
                    # Draw the chessboard corners
                    cv2.drawChessboardCorners(gray2, chessboardSize, corners2, ret)

            # Display the resulting frame
            cv2.imshow('Chessboard capture',gray2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            print('Cannot get more frame from the camera device')
            break
        
        # Update the time elapse
        timeCounter = time.time() - start

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parameters
    DEVICE = 0
    CAPTURES = 4
    INTERVAL = 10 #seconds
    CHESSBOARD_SIZE = (7, 7)
    WIN_SIZE = (5, 5)
    SHOW_CHESSBOARD_CORNERS = True
    CAPTURE_DIRECTORY = './'
    CAPTURE_FILENAME = 'chessboard'
    
    # Setup the command line arguments
    parser = argparse.ArgumentParser(description='Capture the chessboard images')
    parser.add_argument('--device', help='Camera device id (Default is 0)', default=0, type=int)
    parser.add_argument('--captures', help='Number of image to capture (Default is 4)', default=4, type=int)
    parser.add_argument('--interval', help='Interval value in seconds between captures (Default is 10s)', default=10, type=int)
    parser.add_argument('--chessboardSize', help='Size of the chessboard (Default (7, 7))', nargs='+', type=int)
    parser.add_argument('--showChessboardCorner', help='Display the corners detected (Default True)', default=True)
    parser.add_argument('--captureDirectory', help='Path to the directory to store the captures (Default is ./)', default='./')
    parser.add_argument('--captureFilename', help='Capture base filename', default='chessboard')
    args = parser.parse_args()
    
    DEVICE = int(args.device)
    CAPTURES = int(args.captures)
    INTERVAL = int(args.interval)
    SHOW_CHESSBOARD_CORNERS = bool(args.showChessboardCorner)
    CAPTURE_DIRECTORY = str(args.captureDirectory)
    CAPTURE_FILENAME = str(args.captureFilename)
    if args.chessboardSize != None:
        CHESSBOARD_SIZE = tuple(args.chessboardSize)
    
    # Start the capture process
    captureChessboard(DEVICE, CAPTURES, INTERVAL, CHESSBOARD_SIZE, WIN_SIZE, SHOW_CHESSBOARD_CORNERS, CAPTURE_DIRECTORY, CAPTURE_FILENAME)

if __name__ == "__main__":
    main()
