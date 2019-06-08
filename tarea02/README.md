# Segmentation Algorithms with Python
In this folder you can find three different of segmentation algorithms implemented by using Python 3. 
* Mean Shift
* Watershed
* CAM Shift

The developed applications were tested on Linux Ubuntu 16.04 LTS.

#### Requirements:
* [Python 3](https://www.python.org/downloads/)
* [OpenCV 3](https://pypi.org/project/opencv-python/)

#### Usage
* Give executable permissions to the _start_seg.sh_ script:
    ```
    chmod +x start_seg.sh
    ```
* Read the instructions printed by the script:
    ```
    ./start_seg.sh --help
    ```
    The following information will be displayed>
    ```
    ---------------------------------------------------------
    - Main Segmentation Script: start_seg.sh                -
    ---------------------------------------------------------
    
    Usage:
    -h,--help: Prints this help message.
    --ms: Starts the Mean-shift segmentation algorithm.
    --ws: Starts the Watersheds segmentation algorithm.
    --cs: Starts the CAM-shift segmentation algorithm.

    Important considerations:
    - The algorithms are applied upon a camera video stream.
    - The initial parameters can be modified in the file:
        <algorithm>Serialization.py

    ---------------------------------------------------------
    ```
* Modify the initial parameters of the chosen algorithm if you consider it necessary.
* Run the _start_seg.sh_ script indicating the desired algorithm to execute.
