#!/bin/bash

print_help() {
	echo "-------------------------------------------------------------------------------------------------------------"
	echo "-                          Camera Calibration Script: start_calibration.sh                                  -"
	echo "-------------------------------------------------------------------------------------------------------------"
	echo ""
	echo "Usage:"
	echo "-h,--help: Prints this help message."
	echo "--live: Starts the minimal Virtual Reality Demo."
	echo "--ctest: Starts the desired test in order to get an error graph."
	echo "     0 : Use an initialization matrix to get the parameters."
	echo "     1 : Get the parameters with the CALIB_FIX_PRINCIPAL_POINT flag."
	echo "     2 : Get the parameters without an initialization matrix."
	echo "     3 : Use different number of captured images in a range from 1 to 10."
	echo ""
	echo "Important considerations:"
	echo " - The matrix of intrinsic/extrinsic parameters is stored in a JSON format file located in the matrix_params/ folder."
	echo " - The error values are written in a text file located in the results/ folder along with the PNG plots."
	echo " - The generated JSON files are located in the matrix_params/ folder."
	echo " - The text files and the JSON files have a timestamp as part of their names."
	echo " - The minimal VR Demo will use a default matrix-parameters JSON file located in the ./matrix_params/default/ folder."
	echo ""
	echo " - You can generate your own captures by using the ./src/captureChessboardImages.py script as follows: "
	echo "      # python3 captureChessboardImages.py --chessboardSize [width] [height] --captureDirectory [your_dir] --captures [n_images] --interval [interval seconds]"
	echo " - For more information run:  # python3 captureChessboardImages.py --help."
	echo ""
	echo " - You can also generate a JSON file with the matrix parameters by using the ./src/calibrationTest.py script as follows: "
	echo "      # python3 calibrationTest.py --calibrateFilename ../matrix_params/[your_params].json"
	echo " - For more information run:  # python3 calibrationTest.py --help."
	echo ""
	echo " - The tests from 0 to 2 use ten sets of nine captured images each, to get the error graph."
	echo "-------------------------------------------------------------------------------------------------------------"
}

if test "$#" -lt 1; then
    echo "ERROR: Illegal number of parameters"
elif [[ "$1" == @(-h|--help) ]]; then
	print_help
elif [[ "$1" != @(--live|--ctest) ]]; then
	echo "ERROR: Invalid argument $1"
else
	case "$1" in
     "--live")
          echo "Minimal Virtual Reality Demo selected!"
          echo "Using default matrix parameters"
          cd src/
          python3 calibrationTest.py --calibrateFilename ../matrix_params/default/calibration-1564331903.984122.json
          cd ..
          ;; 
     "--ctest")
          echo "Calibration Error Test selected!"
          echo "Using pre-captured set of images"
        	case "$2" in
     		  "0")
          		  echo "Using an initialization matrix to get the parameters..."
          		  cd src/
          		  python3 cameraCalibration.py --testid 0
          		  cd ..
          		  ;; 
     		  "1")
          		  echo "Getting the matrix parameters with the CALIB_FIX_PRINCIPAL_POINT flag..."
          		  cd src/
          		  python3 cameraCalibration.py --testid 1
          		  cd ..
          		  ;;
          	  "2")
          		  echo "Getting the parameters without an initialization matrix..."
          		  cd src/
          		  python3 cameraCalibration.py --testid 2
          		  cd ..
          		  ;; 
     		  "3")
          		  echo "Using different number of captured images in a range from 1 to 10..."
          		  cd src/
          		  python3 cameraCalibration.py --testid 3
          		  cd ..
          		  ;;
    		esac
    esac
fi
