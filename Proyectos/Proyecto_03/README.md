# Detección de Anomalías mediante Aprendizaje Profundo

## Estructura del Proyecto

En esta carpeta se pueden encontrar los siguientes archivos y subcarpetas:

```
.
├── build_dataset.py
├── networks
│   ├── config.py
│   └── variational_autoencoder
│       ├── anomaly_detector.py
│       ├── roc.png
│       ├── utils.py
│       ├── vae_cnn.h5
│       ├── vae_cnn_model.py
│       ├── vae_cnn.png
│       ├── vae_mlp_decoder.png
│       ├── vae_mlp_encoder.png
│       ├── vae_mlp_error.png
│       ├── vae_mlp.h5
│       ├── vae_mlp.png
│       ├── vae_model.py
│       └── vae.py
├── gen_data
│   └── gen_data.py
└── README.md

```

* __networks/config.py:__ Script donde se configuran las rutas, parámetros e hiperparámetros de las Redes Neuronales a utilizar.

* __networks/variational_autoencoder/anomaly_detector.py:__ Script que contiene una clase para evaluar si en las imágenes se tienen anomalías o no, permitiendo además la segmentación de dichas anomalías en caso de existir.

* __networks/variational_autoencoder/utils.py:__ Script que contiene un conjunto de funciones para preprocesar el set de datos y para mejorar el reporte y graficación de los resultados.

* __networks/variational_autoencoder/vae_model.py:__ Script donde se definen las operaciones de un Autoencoder Variacional que utiliza una Red Neuronal Convolucional (CNN). Las operaciones permiten entrenar y probar los parámetros de red utilizados para procesar un set de datos definido por el usuario. En este caso se cuenta con dos set de datos a elegir.

* __networks/variational_autoencoder/vae.py:__ Script donde se definen las operaciones de un Autoencoder Variacional que utiliza un Perceptrón Multicapa (MLP). Las operaciones permiten entrenar y probar los parámetros de red utilizados para procesar un set de datos definido por el usuario.

* __networks/variational_autoencoder/*.png__: Imágenes obtenidas para visualizar el rendimiento de las redes y su estructura.

* __networks/variational_autoencoder/*.h5__: Modelos productos del proceso de entrenamiento.

* __build_dataset.py:__ Script utilizado para separar el set de datos en los conjuntos de entrenamiento, validación y prueba.

* __gen_data/gen_data.py:__ Script utilizado para generar un set de datos de 10044 imágenes con elipsoides en un espacio tridimensional aleatorio.


## Set de Datos:

Los set de datos utilizados en este proyecto son los siguientes:

### Set de Datos 01 - Cáncer de Mama

Set de datos cuya información se encuentra en [este enlace](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). Las imágenes de este set son cortadas en imágenes más pequeñas de 50x50 para generar más de 100 000 imágenes.

* Total de Imágenes: 569
* Imágenes Normales: 357
* Imágenes con Anomalías: 212
* [Enlace de Acceso](https://www.dropbox.com/s/hw326x88leq7voz/IDC_regular_ps50_idx5.zip?dl=0)


### Set de Datos 02 - Elipsoides:

Corresponde a un conjunto de imágenes sintéticas de 360x290, fabricadas con el script denominado __gen_data.py__.

* Total de Imágenes: 10044
* Imágenes Normales: 9300
* Imágenes con Anomalías: 744
* [Enlace de Acceso](https://www.dropbox.com/s/8xzpfbq9b8ws6mh/dataset.tar.gz?dl=0)


## Instrucciones de Uso:

* Entrenamiento de una CNN con el dataset seleccionado: [dataset_id] = 1 ó 2

```
python vae.py --cnn -ds [dataset_id]
```

* Entrenamiento de un MLP con el dataset seleccionado y una función de error MSE:

```
python vae.py --mse -ds [dataset_id]
```

* Predicción sobre una imagen en particular utilizando la red MLP indicando los pesos pre-entrenados:

```
python vae.py --mse -ds [dataset_id] -w vae_mlp.h5 --predict ../../datasets/idc/testing/1/[image_name].png
```

* Predicción sobre una imagen en particular utilizando la red CNN indicando los pesos pre-entrenados:

```
python vae.py --cnn -ds [dataset_id] -w vae_cnn.h5 --predict ../../datasets/idc/testing/1/[image_name].png
```
