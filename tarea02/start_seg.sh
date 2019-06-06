#!/bin/bash

print_help() {
	echo "------------------------------------------------------"
	echo "- Main Segmentation Script                           -"
	echo "------------------------------------------------------"
	echo ""
	echo "Usage:"
	echo "-h,--help: Prints this help message."
	echo "--ms: Starts the Mean-shift segmentation algorithm."
	echo "--ws: Starts the Watersheds segmentation algorithm."
	echo "--cs: Starts the CAM-shift segmentation algorithm."
	echo ""
	echo "The algorithms are applied upon a camera video stream."
	echo ""
	echo "------------------------------------------------------"
}

if test "$#" -ne 1; then
    echo "ERROR: Illegal number of parameters"
elif [[ "$1" == @(-h|--help) ]]; then
	print_help
elif [[ "$1" != @(--ms|--ws|--cs) ]]; then
	echo "ERROR: Invalid argument $1"
else
	echo "Good Parameter!"
fi