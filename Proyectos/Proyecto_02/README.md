# Calibración de cámara

## Estructura del Proyecto

En esta carpeta se pueden encontrar los siguientes archivos y carpetas:

```
.
├── matrix_params
│   └── default
│       └── calibration-1564331903.984122.json
├── README.md
├── results
├── src
│   ├── calibrationTest.py
│   ├── cameraCalibration.py
│   └── captureChessboardImages.py
├── start_calibration.sh
└── test_dirs
    ├── test_dir1
    ├── test_dir10
    ├── test_dir2
    ├── test_dir3
    ├── test_dir4
    ├── test_dir5
    ├── test_dir6
    ├── test_dir7
    ├── test_dir8
    ├── test_dir9
    ├── test_set1
    ├── test_set10
    ├── test_set2
    ├── test_set3
    ├── test_set4
    ├── test_set5
    ├── test_set6
    ├── test_set7
    ├── test_set8
    └── test_set9
```

* __matrix_params:__ Carpeta que contiene los parámetros intrínsecos y de distorción que se utilizan por defecto, en caso de no ser estos especificados o generados previamente.

* __results:__ Carpeta que contiene los resultados de las calibraciones, con gráficas que los representan.

* __src:__ Carpeta que contiene los 3 módulos escritos en pyton utilizados en el proceso de calibración.

* __start_calibration.sh:__ Script central desdel el cual es posible ejecutar los diferentes módulos de calibración, así como diferentes pruebas pre-establecidas.

* __test_dirs:__ Carpeta que contiene diversos conjuntos de capturas utilizados por las pruebas planteadas para la calibración.


## Módulos:

* __captureChessboardImages.py:__ Módulo encargado de la captura del patron planar, en este caso de un tablero de ajedrez.

* __cameraCalibration.py:__ Módulo que efectua el cálculo de los parámetros intrínsecos de la cámara.

* __calibrationTest.py:__ Módulo que, bajo ciertos parámetros internos, calcula los parámetros extrínsecos y proyecta un eje de coordenadas sobre un vide en vivo de la cámara.


## Requerimientos del Sistema:

### Bibliotecas:

* [Python 3](https://www.python.org/downloads/)

* [OpenCV 3.4.1](https://opencv.org/opencv-3-4-1/)

## Instrucciones de Uso:

### Script __start_calibration.sh__:

```
    $ ./start_calibration.sh --help
    -------------------------------------------------------------------------------------------------------------
    -                          Camera Calibration Script: start_calibration.sh                                  -
    -------------------------------------------------------------------------------------------------------------

    Usage:
    -h,--help: Prints this help message.
    --live: Starts the minimal Virtual Reality Demo.
    --ctest: Starts the desired test in order to get an error graph.
        0 : Use an initialization matrix to get the parameters.
        1 : Get the parameters with the CALIB_FIX_PRINCIPAL_POINT flag.
        2 : Get the parameters without an initialization matrix.
        3 : Get the parameters with the CALIB_FIX_ASPECT_RATIO flag.
        4 : Use different number of captured images in a range from 3 to 10.

    Important considerations:
    - The matrix of intrinsic/extrinsic parameters is stored in a JSON format file located in the matrix_params/ folder.
    - The error values are written in a text file located in the results/ folder along with the PNG plots.
    - Each plot shows 4 different graphs: mean error, median error, standard deviation error and the highest error value.
    - The generated JSON files are located in the matrix_params/ folder.
    - The text files and the JSON files have a timestamp as part of their names.
    - The minimal VR Demo will use a default matrix-parameters JSON file located in the ./matrix_params/default/ folder.

    - You can generate your own captures by using the ./src/captureChessboardImages.py script as follows: 
        # python3 captureChessboardImages.py --chessboardSize [width] [height] --captureDirectory [your_dir] --captures [n_images] --interval [interval seconds]
    - For more information run:  # python3 captureChessboardImages.py --help.

    - You can also generate a JSON file with the matrix parameters by using the ./src/calibrationTest.py script as follows: 
        # python3 calibrationTest.py --calibrateFilename ../matrix_params/[your_params].json
    - For more information run:  # python3 calibrationTest.py --help.

    - The tests from 0 to 2 use ten sets of nine captured images each, to get the error graph.
    -------------------------------------------------------------------------------------------------------------

```

### Script __captureChessboardImages.py__:

    ```
    $ python captureChessboardImages.py --help
    usage: captureChessboardImages.py [-h] [--device DEVICE] [--captures CAPTURES]
                                      [--interval INTERVAL]
                                      [--chessboardSize CHESSBOARDSIZE [CHESSBOARDSIZE ...]]
                                      [--showChessboardCorner SHOWCHESSBOARDCORNER]
                                      [--captureDirectory CAPTUREDIRECTORY]
                                      [--captureFilename CAPTUREFILENAME]

    Capture the chessboard images

    optional arguments:
      -h, --help            show this help message and exit
      --device DEVICE       Camera device id (Default is 0)
      --captures CAPTURES   Number of image to capture (Default is 4)
      --interval INTERVAL   Interval value in seconds between captures (Default is 10s)
      --chessboardSize CHESSBOARDSIZE [CHESSBOARDSIZE ...]
                            Size of the chessboard (Default (7, 7))
      --showChessboardCorner SHOWCHESSBOARDCORNER
                            Display the corners detected (Default True)
      --captureDirectory CAPTUREDIRECTORY
                            Path to the directory to store the captures (Default is ./)
      --captureFilename CAPTUREFILENAME
                            Capture base filename
    ```

### Script __cameraCalibration.py__:
    
    ```
    $ python cameraCalibration.py --help
    usage: cameraCalibration.py [-h]
                                [--chessboardSize CHESSBOARDSIZE [CHESSBOARDSIZE ...]]
                                [--captureDirectory CAPTUREDIRECTORY]
                                [--testid TESTID]

    Camera calibration process

    optional arguments:
      -h, --help            show this help message and exit
      --chessboardSize CHESSBOARDSIZE [CHESSBOARDSIZE ...]
                            Size of the chessboard (Default (7, 7))
      --captureDirectory CAPTUREDIRECTORY
                            Path to the directory to store the captures (Default is ./)
      --testid TESTID       ID of the test to be performed

    ```

### Script __calibrationTest.py__:

    ```
    $ python calibrationTest.py --help
    usage: calibrationTest.py [-h] [--device DEVICE]
                              [--calibrateFilename CALIBRATEFILENAME]
                              [--chessboardSize CHESSBOARDSIZE [CHESSBOARDSIZE ...]]

    Test script for the camera calibration

    optional arguments:
      -h, --help            show this help message and exit
      --device DEVICE       Camera device id (Default is 0)
      --calibrateFilename CALIBRATEFILENAME
                            Calibration config filename
      --chessboardSize CHESSBOARDSIZE [CHESSBOARDSIZE ...]
                            Size of the chessboard (Default (7, 7))

    ```
