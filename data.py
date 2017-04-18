import numpy as np
import cv2
import random
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle

def preprocess(image, convert_YUV=True):
    """
    Preprocess provided image, optionally converting to YUV color
    space, cropping top and bottom of image, and rescaling to match
    resolution of Nvidia convolutional network input size.

    :returns: a preprocessed image
    """
    if convert_YUV == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = image[40:146, 0:320]
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_CUBIC) 
    return image

def data_frames_from_paths(paths):
    """
    Returns a pandas DataFrame containing the data represented in the paths
    parameter.

    :param paths: the paths at which to find driving_log.csv files and associated images
    :returns: data frames associated with the driving_log.csv files in
        the provided paths
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path + 'driving_log.csv', header=None)
        path_column = pd.DataFrame({
            'path': [path] * df.shape[0]
        })
        frames.append(df.join(path_column))
    df = pd.concat(frames)
    df.columns = [
        'center', 'left', 'right', 'steering',
        '', '', '', 
        'path',
    ]
    return df

def balanced_samples(
    samples, 
    num_bins=100,
    histogram_equalization_factor=2
):
    """
    Returns a collection of pandas DataFrame objects, balanced by sampling
    the entire set so that steering measurements are more uniformly represented.

    :param samples: the unbalanced samples, a collection of pandas DataFrame objects
    :param num_bins: the number of bins to use to generate and then equalize the histogram
        of samples based on steering angle
    :param histogram_equalization_factor: a factor that adjusts how aggressively the 
        steering angle histogram is balanced by sampling the wholedataset
    :returns: balanced_samples, a collection of pandas DataFrame objects that represent
        a subset of the data, balanced to some extent based on steering angle
    """
    balanced = pd.DataFrame()
    max_bin_count = int(samples.shape[0] / num_bins / histogram_equalization_factor)

    min_steering = min(samples['steering'])
    max_steering = max(samples['steering'])

    start = min_steering
    end_space = np.linspace(start, max_steering, num=num_bins)
    for end in end_space:
        indexes = samples[
            (samples.steering >= start) &
            (samples.steering < end)
        ]
        if indexes.shape[0] == 0:
            continue
        range_n = min(max_bin_count, indexes.shape[0])
        balanced = pd.concat([balanced, indexes.sample(range_n)])
        start = end
    return balanced

def generator(samples, batch_size, steering_adjust_factor=0.15, convert_YUV=True):
    """
    A generator function that yields a subset of data (images and
    steering angles) every time it is invoked.

    :param samples: the samples (pandas DataFrame objects) to draw
        from to return a value
    :param batch_size: the size of the batch and number of images and
        steering values to return
    :param steering_adjust_factor: the absolute value of a steering
        adjust value to apply to off-center images found in samples
    :returns: a tuple of (X, y) numpy arrays, each containing a subset
        of the data. The length of the numpy arrays is 3 * batch_size,
        larger than batch_size because left, right, and center camera
        images are used for each sample in the batch_size number of
        samples
    """
    num_samples = samples.shape[0]
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            steerings = []
            for (index, row) in batch_samples.iterrows():
                views = ['center', 'left', 'right']
                adjustments = [
                    0, 
                    steering_adjust_factor, 
                    -steering_adjust_factor
                ]
                for view, adjustment in zip(views, adjustments):
                    filename = row[view].split('/')[-1]
                    path = row['path'] + 'IMG/' + filename
                    image = preprocess(cv2.imread(path), convert_YUV=convert_YUV)
                    steering = float(row['steering']) + adjustment
                    images.append(image)
                    steerings.append(steering)
            X = np.array(images)
            y = np.array(steerings)

            # Flip a random sample of our images, reversing the
            # ground truth steering angle.
            X_len = X.shape[0]
            flip_indices = random.sample(range(X_len), int(X_len / 8))
            X[flip_indices] = X[flip_indices, :, ::-1, :]
            y[flip_indices] = -y[flip_indices]
            yield X, y