#!/bin/bash

print_help() {
	echo "---------------------------------------------------------"
	echo "- Main Segmentation Script: start_seg.sh                -"
	echo "---------------------------------------------------------"
	echo ""
	echo "Usage:"
	echo "-h,--help: Prints this help message."
	echo "--ms: Starts the Mean-shift segmentation algorithm."
	echo "--ws: Starts the Watersheds segmentation algorithm."
	echo "--fw: Starts the Felzenszwalb segmentation algorithm."
	echo ""
	echo "Important considerations:"
	echo " - The algorithms are applied upon a camera video stream."
	echo " - The initial parameters can be modified in the file:"
	echo "   <algorithm>Serialization.py"
	echo " - For the case of the mean shift and felzenszwalb algorithms"
	echo "   modify the parameters stored in the JSON files"
	echo ""
	echo "---------------------------------------------------------"
}

if test "$#" -ne 1; then
    echo "ERROR: Illegal number of parameters"
elif [[ "$1" == @(-h|--help) ]]; then
	print_help
elif [[ "$1" != @(--ms|--ws|--cs) ]]; then
	echo "ERROR: Invalid argument $1"
else
	case "$1" in
     "--ms")
          echo "Mean-shift algorithm selected!"
          cd meanShift/
          python3 meanShift.py
          cd ..
          ;; 
     "--ws")
          echo "Watershed algorithm selected!"
          cd watershed/
          python3 watershedSerialization.py
          python3 watershed.py
          cd ..
          ;;
     "--fw")
          echo "Felzenszwalb algorithm selected!"
          cd felzenszwalb/
          python3 felzenszwalb.py
          cd ..
          ;; 
    esac
fi
