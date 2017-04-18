from pandas import DataFrame
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from random import randint
matplotlib.use('Agg') 
matplotlib.style.use('ggplot')

from data import generator
from data import balanced_samples
from data import data_frames_from_paths

def plot_sample_frames_simple_coures():
    PATHS = [
        './Driving_Data/16-April/',
    ]
    plot_sample_frames(PATHS, 'simple_course_examples.png')

def plot_sample_frames_hard_course():
    PATHS = [
        './Driving_Data/16-April_Hard/',
    ]
    plot_sample_frames(PATHS, 'jungle_course_examples.png')

def plot_sample_frames(paths, figure_name='images.png'):
    df = balanced_samples(
        data_frames_from_paths(paths),
        histogram_equalization_factor=1.5
    )

    images, steering = next(generator(df, batch_size=2048, convert_YUV=False))

    fig = plt.figure()
    for i in range(9):
        if i == 0:
            index = np.where(steering > 0.03)[0][0]
        else:
            index = randint(0, 3 * 2048)
        ax = fig.add_subplot(331 + i, xticks=[], yticks=[])
        ax.imshow(cv2.cvtColor(images[index], cv2.COLOR_BGR2RGB))
        annotation = "{:.2f}".format(steering[index])
        plt.text(30 + i / 3 * 10, 30 + i % 3 * 10, annotation, fontsize=8, color='purple', backgroundcolor='white')

    plt.tight_layout()
    fig.savefig(figure_name)

def plot_steering_angle_histograms(bins=100):
    PATHS = [
        './Driving_Data/16-April/',
        './Driving_Data/16-April_Reverse/',
        './Driving_Data/16-April_Hard/',
    ]
    df = data_frames_from_paths(PATHS)
    fig = plt.figure()
    ax = plt.subplot(111)
    DataFrame.hist(df, ax=ax, column='steering', bins=bins)
    plt.title('Unbalanced Dataset Steering Angles')
    plt.ylabel('Number')
    plt.xlabel('Steering Value')
    fig.savefig('unbalanced_steering.png')

def plot_balanced_steering_angle_histograms(bins=100):
    PATHS = [
        './Driving_Data/16-April/',
        './Driving_Data/16-April_Reverse/',
        './Driving_Data/16-April_Hard/',
    ]
    df = balanced_samples(
        data_frames_from_paths(PATHS),
        histogram_equalization_factor=1.5
    )
    fig = plt.figure()
    ax = plt.subplot(111)
    DataFrame.hist(df, ax=ax, column='steering', bins=bins)
    plt.title('Balanced Dataset Steering Angles')
    plt.ylabel('Number')
    plt.xlabel('Steering Value')
    fig.savefig('balanced_steering.png')


def plot_loss(history_dict):
    """
    Create and save a plot demonstrating the loss over time for
    training and validation sets, given a Keras dictionary of
    fit results (e.g. the return value from `model.fit_generator`)
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(history_dict['loss'])
    ax.plot(history_dict['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    ax.legend(['Train', 'Validation'], loc='upper right')
    fig.savefig('accuracy.png')
