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
    help_ = "Dataset ID: 1-Breast Cancer Set (default), 2-Textured Ellipsoids"
    parser.add_argument("-ds",
                        "--dataset",
                        help=help_,
                        type=int,
                        default=1)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_,action='store_true')
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
    dataset_id = int(args.dataset)
    image_width = 50 if dataset_id == 1 else 360
    image_height = 50 if dataset_id == 1 else 290
    original_dim = image_width * image_height

    # VAE model = encoder + decoder
    vae = None
    if args.cnn:
        inputShape = (image_width - 2, image_height - 2, 3)
        vae = VAECNN(input_shape=inputShape, latent_cont_dim=8, latent_disc_dim=3)
    else:
        inputShape = (image_width - 2) * (image_height - 2)
        vae = VAE(inputShape, args.batch, args.epochs, image_width, image_height)
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
        genPath = config.TRAIN_CANCER_PATH if dataset_id == 1 else config.TRAIN_ELLIPS_PATH
        trainGen = trainAug.flow_from_directory(
        	os.path.sep.join([config.NET_BASE, genPath]),
        	class_mode="input",
            target_size=(image_width - 2, image_height - 2),
        	color_mode="rgb",
        	shuffle=True,
        	batch_size=args.batch)

        # initialize the validation generator
        genPath = config.VAL_ELLIPS_PATH if dataset_id == 1 else config.VAL_ELLIPS_PATH
        valGen = valAug.flow_from_directory(
        	os.path.sep.join([config.NET_BASE, genPath]),
        	class_mode="input",
        	target_size=(image_width - 2, image_height - 2),
        	color_mode="rgb",
        	shuffle=False,
        	batch_size=args.batch)

        # initialize the testing generator
        genPath = config.TEST_ELLIPS_PATH if dataset_id == 1 else config.TEST_ELLIPS_PATH
        testGen = valAug.flow_from_directory(
        	os.path.sep.join([config.NET_BASE, genPath]),
        	class_mode="input",
        	target_size=(image_width - 2, image_height - 2),
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
        x_train, x_val = utils.getData(img_width=image_width, img_height=image_height, dataset_id=dataset_id)
        print("Dataset loaded")
        print("Start training...")
        vae.train(x_train, x_val)

    if (args.plot):
        vae.plot()

    if predict_img != '':
        img = cv2.imread(predict_img)
        orig = img
        #img = utils.preprocess(img, image_width, image_height)
        #images = np.array([img])
        reconstruction_error, ssim, rec = vae.prediction(img)
        print("Reconstruction error: " + str(reconstruction_error))

        detector = AnomalyDetector(anomaly_treshold = 0.45)
        detector.evaluate(reconstruction_error, ssim, orig, rec, dataset_id)

    if args.test:

        y_prob = []
        y_res = []
        normal_res = []
        anormal_res = []

        patients = os.listdir(os.path.sep.join([config.NET_BASE, config.ORIG_INPUT_CANCER_DATASET]))
        random.seed(3)
        random.shuffle(patients)

        for patient in patients:#patients[-3:]:
            x_normal, x_anormal = utils.getValData(patient, image_width, image_height, dataset_id)

            y_prob += np.zeros(len(x_normal)).tolist() + np.ones(len(x_anormal)).tolist()
            for normal_img in x_normal:
                reconstruction_error, ssim, _ = vae.prediction(normal_img)
                normal_res.append(reconstruction_error)
                if reconstruction_error < 40 and ssim > 0.4 :
                    y_res.append(0)
                else:
                    y_res.append(1)

            for anormal_img in x_anormal:
                reconstruction_error, ssim,  _ = vae.prediction(anormal_img)
                anormal_res.append(reconstruction_error)
                if reconstruction_error < 40 and ssim > 0.4 :
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
