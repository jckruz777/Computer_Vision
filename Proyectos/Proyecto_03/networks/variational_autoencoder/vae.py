from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os

from keras.preprocessing.image import ImageDataGenerator
from anomaly_detector import AnomalyDetector
from sklearn.metrics import roc_curve, auc
from vae_cnn_model import VAECNN
from vae_model import VAE
import random
import utils

import sys
sys.path.append('..')
import config
from imutils import paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    help_ = "Epochs value (default = 50)"
    parser.add_argument("--epochs",
                        help=help_,
                        type=int,
                        default=50)
    help_ = "Batch value (default = 128)"
    parser.add_argument("--batch",
                        help=help_,
                        type=int,
                        default=128)
    help_ = "Use CNN VAE"
    parser.add_argument("--cnn",
                        help=help_,
                        action='store_true')
    help_ = "Test the breast cancer dataset"
    parser.add_argument("--test",
                        help=help_,
                        action='store_true')
    help_ = "Enable the plot feature"
    parser.add_argument("-p",
                        "--plot",
                        help=help_, action='store_true')
    help_ = "Predict image reconstruction"
    parser.add_argument("--predict",
                        help=help_,
                        default='')
    args = parser.parse_args()

    predict_img = str(args.predict)
    image_size = 50
    original_dim = image_size * image_size

    # VAE model = encoder + decoder
    vae = None
    if args.cnn:
        vae = VAECNN(latent_cont_dim=8)
    else:
        vae = VAE(original_dim, args.batch, args.epochs)
        vae.build()

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = "mse" if args.mse else "binary_crossentropy"
        vae.setReconstructionError(reconstruction_loss)

        # Compile the model
        vae.compile()

    if args.weights:
        vae.loadWeights(args.weights)
    elif args.cnn:

        # train the autoencoder
        print("Loading the dataset...")

        # initialize the training data augmentation object
        trainAug = ImageDataGenerator(
        	rescale=1 / 255.0,
        	rotation_range=20,
        	zoom_range=0.05,
        	width_shift_range=0.1,
        	height_shift_range=0.1,
        	shear_range=0.05,
        	horizontal_flip=True,
        	vertical_flip=True,
        	fill_mode="nearest")
            #preprocessing_function=imageResize)

        # initialize the validation (and testing) data augmentation object
        valAug = ImageDataGenerator(rescale=1 / 255.0)
                                    #preprocessing_function=imageResize)

        # initialize the training generator
        trainGen = trainAug.flow_from_directory(
        	os.path.sep.join([config.NET_BASE, config.TRAIN_PATH]),
        	class_mode="input",
        	target_size=(48, 48),
        	color_mode="rgb",
        	shuffle=True,
        	batch_size=args.batch)

        # initialize the validation generator
        valGen = valAug.flow_from_directory(
        	os.path.sep.join([config.NET_BASE, config.VAL_PATH]),
        	class_mode="input",
        	target_size=(48, 48),
        	color_mode="rgb",
        	shuffle=False,
        	batch_size=args.batch)

        # initialize the testing generator
        testGen = valAug.flow_from_directory(
        	os.path.sep.join([config.NET_BASE, config.TEST_PATH]),
        	class_mode="input",
        	target_size=(48, 48),
        	color_mode="rgb",
        	shuffle=False,
        	batch_size=args.batch)

        #x_train, x_val = utils.getData(nd_images=True)
        print("Dataset loaded")
        print("Start training...")
        vae.fit(trainGen, valGen, num_epochs=args.epochs, batch_size=args.batch)
    else:
        # train the autoencoder
        print("Loading the dataset...")
        x_train, x_val = utils.getData()
        print("Dataset loaded")
        print("Start training...")
        vae.train(x_train, x_val)

    if (args.plot):
        vae.plot()

    if predict_img != '':
        img = cv2.imread(predict_img)
        orig = img
        img = utils.preprocess(img)
        images = np.array([img])
        reconstruction_error, rec = vae.prediction(images)
        print("Reconstruction error: " + str(reconstruction_error))

        detector = AnomalyDetector(anomaly_treshold = 60)
        detector.evaluate(reconstruction_error, orig, rec)

    if args.test:

        y_prob = []
        y_res = []
        normal_res = []
        anormal_res = []

        patients = os.listdir(os.path.sep.join([config.NET_BASE, config.ORIG_INPUT_DATASET]))
        random.seed(3)
        random.shuffle(patients)

        for patient in patients[-3:]:
            x_normal, x_anormal = utils.getValData(patient)

            y_prob += np.zeros(len(x_normal)).tolist() + np.ones(len(x_anormal)).tolist()
            for normal_img in x_normal:
                images = np.array([normal_img])
                reconstruction_error, _ = vae.prediction(images)
                normal_res.append(reconstruction_error)
                if reconstruction_error < 40:
                    y_res.append(0)
                else:
                    y_res.append(1)

            for anormal_img in x_anormal:
                images = np.array([anormal_img])
                reconstruction_error, _ = vae.prediction(images)
                anormal_res.append(reconstruction_error)
                if reconstruction_error < 40:
                    y_res.append(0)
                else:
                    y_res.append(1)

        fpr, tpr, thresholds = roc_curve(y_prob, y_res)

        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig('roc.png')
        plt.show()

        print("Average error for normal images: " + str(np.average(np.array(normal_res))))
        print("Average error for anomal images: " + str(np.average(np.array(anormal_res))))

        plt.scatter(range(len(normal_res)), normal_res)
        plt.scatter(range(len(anormal_res)), anormal_res)
        plt.title('Reconstruction test')
        plt.ylabel('Loss')
        plt.xlabel('Image')
        plt.legend(['Normal', 'Anomaly'], loc='upper right')
        plt.savefig('reconstruction_test.png')
        plt.show()

    # if args.plot:
    #     utils.plot_results(models,
    #                 data,
    #                 batch_size=batch_size,
    #                 model_name="vae_mlp")
