import simplejson
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

from data import generator
from data import balanced_samples
from data import data_frames_from_paths

from plots import plot_loss 

NUM_EPOCHS = 8
BATCH_SIZE = 64
MODEL_NAME = "model"

"""
 The number of image, steering pairs provided per sample in a batch. This
 number is greater than 1 because, e.g., we provide three images and steering
 pairs for left, right, and center cameras for a single sample from our
 driving_log file. If we did additional augmentation (e.g. flipping every
 image), this number could be even larger.
"""
SAMPLES_PER_SAMPLE = 3

"""
The magnitude of steering adjustment applied for left/right images
as compared to the center image.
"""
STEERING_ADJUST = 0.15

"""
A factor adjusting how much the dataset is balanced according to steering
angle by sampling
"""
BALANCED_ADJUST = 1.5

PATHS = [
    './Driving_Data/16-April/',
    './Driving_Data/16-April_Reverse/',
    './Driving_Data/16-April_Hard/',
]

samples = balanced_samples(
    data_frames_from_paths(PATHS), 
    histogram_equalization_factor=BALANCED_ADJUST
)

train_samples, validation_samples = train_test_split(
    samples, 
    test_size=0.2
)

train_generator = generator(
    train_samples,
    batch_size=BATCH_SIZE,
    steering_adjust_factor=STEERING_ADJUST
)
validation_generator = generator(
    validation_samples,
    batch_size=BATCH_SIZE,
    steering_adjust_factor=STEERING_ADJUST
)

input_shape = (66, 200, 3)
normalize_lambda = lambda image: image / 255.0 - 0.5

"""
 The Nvidia architecture desecribed in Bojarski et al. 2016 , 'End to End
 Learning for Self-Driving Cars'. In addition to their architecture, we've
 added some Dropout regularization layers and a cropping layer to focus on a
 specific region of the image.
"""
model = Sequential([
    Lambda(normalize_lambda, input_shape=input_shape),
    Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'),
    Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'),
    Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'),
    Convolution2D(64, 3, 3, activation='relu'),
    Convolution2D(64, 3, 3, activation='relu'),
    Flatten(),
    Dropout(0.25),
    Dense(100, activation='relu'),
    Dropout(0.25),
    Dense(50, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples) * SAMPLES_PER_SAMPLE,
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples) * SAMPLES_PER_SAMPLE,
    nb_epoch=NUM_EPOCHS
)

"""
 We had some technical difficulties saving the entire model in an h5 file,
 and so save the model architecture in a .json file and save the weights in
 an .h5 file.
"""
model_json = model.to_json()
with open(MODEL_NAME + '.json', 'w') as json_file:
    json_file.write(simplejson.dumps(
        simplejson.loads(model_json),
        indent=4
    ))
model.save_weights(MODEL_NAME + '.h5')
model.summary()

plot_loss(history.history)
