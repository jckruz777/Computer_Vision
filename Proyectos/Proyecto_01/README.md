# Rastreo en Imágenes mediante Puntos de Interés

## Estructura del Proyecto

En esta carpeta se pueden encontrar los siguientes archivos y carpetas:

```
.
├── IMG
│   ├── deep_learning_book.jpg
│   ├── test_1.jpg
│   ├── test_2.jpg
│   └── test_3.jpg
├── keypointsStillImageTracking.py
├── keypointsTracking.py
├── project_01.ipynb
└── VIDEO
    ├── vid_01.mp4
    ├── vid_02.mp4
    └── vid_03.mp4
```

* __IMG:__ Carpeta que contiene la imagen de referencia "deep_learning_book.jpg" junto a tres imágenes de prueba.

* __VIDEO:__ Carpeta que contiene tres videos de prueba.

* __project_01.ipynb:__ Cuaderno de Jupyter donde se encuentra el código fuente con texto explicativo.

* __keypointsStillImageTracking.py:__ Script de Python que permite aplicar los algoritmos de detección y rastreo a imágenes estáticas a través de la línea de comandos.

* __keypointsTracking.py:__ Script de Python que permite aplicar los algoritmos de detección y rastreo a videos a través de la línea de comandos.


## Algoritmos Implementados:

Los algoritmos utilizados en este proyecto son los siguientes:

### Detectores:

* __SIFT__: Scale-Invariant Feature Transform

* __SURF__: Speeded-Up Robust Features

* __ORB__: Oriented FAST and Rotated BRIEF

* __BRISK__: Binary Robust Invariant Scalable Keypoints

### _Matchers_:

* __Brute Force__

* __Fast Library for Approximate Nearest Neighbors (FLANN)__

## Requerimientos del Sistema:

### Bibliotecas:

* [Python 3](https://www.python.org/downloads/)

* __TKinter:__ Biblioteca para despliegue gráfico.

```
apt install python3-tk
```

* __OpenCV Contrib:__ Conjunto de bibliotecas que incluye software licenciado de versiones anteriores. Requerido para ejecutar los detectores SIFT y SURF. Se recomienda leer la siguiente guía para obtener este soporte: [Instalación de OpenCV Contrib](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/)

## Instrucciones de Uso:

### Script __keypointsTracking.py__:

* Se puede imprimir un mensaje _help_ con el siguiente contenido:

    ```
    $ python keypointsTracking.py --help
    usage: keypointsTracking.py [-h] [--video VIDEO] [--reference REFERENCE]
                                [--descriptor DESCRIPTOR] [--matcher MATCHER]
                                [--octaves OCTAVES] [--mtreshold MTRESHOLD]

    Keypoints extraction and object tracking

    optional arguments:
      -h, --help            show this help message and exit
      --video VIDEO         Video path
      --reference REFERENCE
                            Image of reference with the object to be detected
      --descriptor DESCRIPTOR
                            Descriptor algorithm: SIFT, SURF, ORB, BRISK
      --matcher MATCHER     Matcher method: Brute force (BF) or FLANN (FLANN)
      --octaves OCTAVES     Number of octaves that the descriptor will use
      --mtreshold MTRESHOLD
                            Treshold for good matches.
    ```

#### Estructura del Comando:

```
python keypointsTracking.py --video <VIDEO_PATH> --reference <IMG_PATH> --descriptor <DESCRIPTOR>
```

Con __DESCRIPTOR__ igual a: __SIFT__, __SURF__, __ORB__ ó __BRISK__.


### Script __keypointsTracking.py__:

* Se puede imprimir un mensaje _help_ con el siguiente contenido:

    ```
    $ python keypointsStillImageTracking.py --help
    usage: keypointsStillImageTracking.py [-h] [--evaluation EVALUATION]
                                          [--reference REFERENCE]
                                          [--descriptor DESCRIPTOR]
                                          [--matcher MATCHER] [--octaves OCTAVES]
                                          [--mtreshold MTRESHOLD]
    Keypoints extraction and object tracking in a still image
    optional arguments:
      -h, --help            show this help message and exit
      --evaluation EVALUATION
                            Image of evaluation
      --reference REFERENCE
                            Image of reference with the object to be detected
      --descriptor DESCRIPTOR
                            Descriptor algorithm: SIFT, SURF, ORB, BRISK
      --matcher MATCHER     Matcher method: Brute force (BF) or FLANN (FLANN)
      --octaves OCTAVES     Number of octaves that the descriptor will use
      --mtreshold MTRESHOLD
                            Treshold for good matches.
    ```

#### Estructura del Comando:

```
python keypointsStillImageTracking.py --evaluation <EVAL_IMG_PATH> --reference <REF_IMG_PATH> --descriptor <DESCRIPTOR>
```

Con __DESCRIPTOR__ igual a: __SIFT__, __SURF__, __ORB__ ó __BRISK__.
