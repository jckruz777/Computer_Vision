import sys
sys.path.append('..')
import config

from keras import backend as K
from imutils import paths

import numpy as np
import time
import cv2
import os


EPSILON = 1e-8

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50))
    return img

def getValData(patient):
    normalPaths = list(paths.list_images(os.path.sep.join([config.NET_BASE, config.ORIG_INPUT_DATASET, patient, "0"])))
    anormalPaths = list(paths.list_images(os.path.sep.join([config.NET_BASE, config.ORIG_INPUT_DATASET, patient, "1"])))

    x_normal = []
    for i, normalImage in enumerate(normalPaths):
        img = cv2.imread(normalImage)
        img = preprocess(img)
        x_normal.append(img)
    x_normal = np.asarray(x_normal)
    #x_normal = np.reshape(x_normal, [-1, 50*50])
    #x_normal = x_normal.astype('float32') / 255

    x_anormal = []
    for i, anormalImage in enumerate(anormalPaths):
        img = cv2.imread(anormalImage)
        img = preprocess(img)
        x_anormal.append(img)
    x_anormal = np.asarray(x_anormal)
    #x_anormal = np.reshape(x_anormal, [-1, 50*50])
    #x_anormal = x_anormal.astype('float32') / 255

    return (x_normal, x_anormal)

def getData(nd_images = False):
    # grab the paths to all input images in the original input directory
    # and shuffle them
    trainPaths = list(paths.list_images(os.path.sep.join([config.NET_BASE, config.TRAIN_PATH])))
    valPaths = list(paths.list_images(os.path.sep.join([config.NET_BASE, config.VAL_PATH])))
    testPaths = list(paths.list_images(os.path.sep.join([config.NET_BASE, config.TEST_PATH])))

    x_train = []
    for i, trainImage in enumerate(trainPaths):
        img = cv2.imread(trainImage)
        img = preprocess(img)
        x_train.append(img)
    x_train = np.asarray(x_train)
    if not nd_images:
        x_train = np.reshape(x_train, [-1, 50*50])
    x_train = x_train.astype('float32') / 255

    x_val = []
    for i, valImage in enumerate(valPaths):
        img = cv2.imread(valImage)
        img = preprocess(img)
        x_val.append(img)
    x_val = np.asarray(x_val)
    if not nd_images:
        x_val = np.reshape(x_val, [-1, 50*50])
    x_val = x_val.astype('float32') / 255

    return (x_train, x_val)

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def get_one_hot_vector(idx, dim=10):
    """
    Returns a 1-hot vector of dimension dim with the 1 at index idx
    Parameters
    ----------
    idx : int
        Index where one hot vector is 1
    dim : int
        Dimension of one hot vector
    """
    one_hot = np.zeros(dim)
    one_hot[idx] = 1.
    return one_hot

def get_timestamp_filename(filename):
    """
    Returns a string of the form "filename_<date>.html"
    """
    date = time.strftime("%H-%M_%d-%m-%Y")
    return filename + "_" + date + ".html"


def kl_normal(z_mean, z_log_var):
    """
    KL divergence between N(0,1) and N(z_mean, exp(z_log_var)) where covariance
    matrix is diagonal.
    Parameters
    ----------
    z_mean : Tensor
    z_log_var : Tensor
    dim : int
        Dimension of tensor
    """
    # Sum over columns, so this now has size (batch_size,)
    kl_per_example = .5 * (K.sum(K.square(z_mean) + K.exp(z_log_var) - 1 - z_log_var, axis=1))
    return K.mean(kl_per_example)


def kl_discrete(dist):
    """
    KL divergence between a uniform distribution over num_cat categories and
    dist.
    Parameters
    ----------
    dist : Tensor - shape (None, num_categories)
    num_cat : int
    """
    num_categories = tuple(dist.get_shape().as_list())[1]
    dist_sum = K.sum(dist, axis=1)  # Sum over columns, this now has size (batch_size,)
    dist_neg_entropy = K.sum(dist * K.log(dist + EPSILON), axis=1)
    return np.log(num_categories) + K.mean(dist_neg_entropy - dist_sum)


def sampling_concrete(alpha, out_shape, temperature=0.67):
    """
    Sample from a concrete distribution with parameters alpha.
    Parameters
    ----------
    alpha : Tensor
        Parameters
    """
    uniform = K.random_uniform(shape=out_shape)
    gumbel = - K.log(- K.log(uniform + EPSILON) + EPSILON)
    logit = (K.log(alpha + EPSILON) + gumbel) / temperature
    return K.softmax(logit)


def sampling_normal(z_mean, z_log_var, out_shape):
    """
    Sampling from a normal distribution with mean z_mean and variance z_log_var
    """
    epsilon = K.random_normal(shape=out_shape, mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon
