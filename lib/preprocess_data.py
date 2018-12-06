import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os


def reshape(images, width):
    '''reshape the input into 32x32x3 np.ndarray'''

    # input images should be a 2-dimensional np.array 
    # e.g: [[1,2,3,...]] for one image only
    first = images.shape[0]
    result = np.zeros((first, width, width, 3))
    index = 0
    for image in images:
        assert len(image) == width * width * 3
        # Get color out of original array
        redPixel = image[0:width * width] / 255
        greenPixel = image[width * width:width * width * 2] / 255
        bluePixel = image[width * width * 2:width * width * 3] / 255
        reshaped = np.zeros((32, 32, 3))
        for i in range(0, width):  # row
            for j in range(0, width):  # column
                point = np.zeros(3)
                point[0] = redPixel[i * 32 + j]
                point[1] = greenPixel[i * 32 + j]
                point[2] = bluePixel[i * 32 + j]
                # add to result
                reshaped[i][j] = point
        result[index] = reshaped
        index += 1

    return result


def plot_images(images, TrueClass, PredClass=None):
    '''show images:
    input np.ndarray, output images with true or predicted class labels'''
    class_names = ['airplane','automobile','bird','cat',
                   'deer','dog','frog','horse','ship','truck']

    assert len(images) == len(TrueClass) == 10

    # Each image as a sub-plot: totally 2 rows 5 columns
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    reshaped_images = reshape(images, 32)

    for i, ax in enumerate(axes.flat):
        # Plot image and smooth it
        ax.imshow(reshaped_images[i],
                  interpolation='spline16')

        # Name of the true class.
        TrueName = class_names[TrueClass[i]]

        if PredClass is None:
            # Only show True class names
            xlabel = "True: {0}".format(TrueName)
        else:
            # Show both True and Pred class names
            PredName = class_names[PredClass[i]]
            xlabel = "True: {0}\nPred: {1}".format(TrueName, PredName)

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def distorted_image(image, cropped_size, training):
    '''This function takes a single image from training set as input'''

    if training:
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[cropped_size, cropped_size, 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

    else:
        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.

        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=cropped_size,
                                                       target_width=cropped_size)

    return image


def preprocess(images, cropped_size, training):
    '''This function takes multiple images as input,
    will call distorted_image()'''

    images = tf.map_fn(lambda image: distorted_image(image, cropped_size, training), images)

    return images


