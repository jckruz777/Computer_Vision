from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os

from anomaly_detector import AnomalyDetector
from vae_cnn_model import VAECNN
from vae_model import VAE
import utils


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
        vae = VAECNN(latent_disc_dim=10)
    else:
        vae = VAE(original_dim, args.batch, args.epochs)
        vae.build()

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = "mse" if args.mse else "binary_crossentropy"
        vae.setReconstructionError(reconstruction_loss)

        # Compile the model
        vae.compile()

    if (args.plot):
        vae.plot()

    if args.weights:
        vae.loadWeights(args.weights)
    elif args.cnn:
        # train the autoencoder
        print("Loading the dataset...")
        x_train, x_val = utils.getData(nd_images=True)
        print(x_train.shape)
        print("Dataset loaded")
        print("Start training...")
        vae.fit(x_train, num_epochs=args.epochs, batch_size=args.batch)
    else:
        # train the autoencoder
        print("Loading the dataset...")
        x_train, x_val = utils.getData()
        print("Dataset loaded")
        print("Start training...")
        vae.train(x_train, x_val)

    if predict_img != '':
        img = cv2.imread(predict_img)
        orig = img
        img = utils.preprocess(img)
        images = np.array([img])
        reconstruction_error, rec = vae.prediction(images)
        print("Reconstruction error: " + str(reconstruction_error))

        detector = AnomalyDetector(anomaly_treshold = 60)
        detector.evaluate(reconstruction_error, orig, rec)

    if args.plot:
        utils.plot_results(models,
                    data,
                    batch_size=batch_size,
                    model_name="vae_mlp")
