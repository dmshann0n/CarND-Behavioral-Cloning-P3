""" Generates a model for use in drive.py """

import csv
import os

from clize import run
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import driving_entry as de

DEFAULT_PATH = "../data_capture"
DEFAULT_MODEL_PATH = "model.h5"

GENERATOR_BATCH_SIZE = 32

OUTSIDE_CAMERA_OFFSET = 0.2

def augment_default_center(entry):
    """ This augmentation returns the unmodified center image and the
        steering angle for that image """

    return entry.center_img(), entry.steering_angle

def augment_flip_image(entry):
    """ This augmentation flips the center image and reverses the steering angle
        to eliminate bias from the left handed track """

    return np.fliplr(entry.center_img()), -entry.steering_angle

def augment_steering_left(entry):
    """ This function returns the left image, and a (hacky) adjustment to the
        steering angle to account for the different perspective from the left """

    return entry.left_img(), entry.steering_angle + OUTSIDE_CAMERA_OFFSET

def augment_steering_right(entry):
    """ This function returns the right image, and a (hacky) adjustment to the
        steering angle to account for the different perspective from the right """
    return entry.right_img(), entry.steering_angle - OUTSIDE_CAMERA_OFFSET

AUGMENTS = [
    augment_default_center,
    augment_flip_image,
    augment_steering_left,
    augment_steering_right
]

NUM_AUGMENTS = len(AUGMENTS)

def build_model(path=None, output_path=None):
    """
    :param path: Path to data dir, defaults to DEFAULT_PATH
    """
    if not path:
        path = DEFAULT_PATH

    if not output_path:
        output_path = DEFAULT_MODEL_PATH

    entries = load_data(path)
    training, validate = train_test_split(entries, test_size=0.2)

    model = Sequential()

    # normalization layers
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        generator=data_generator(training, GENERATOR_BATCH_SIZE),
        steps_per_epoch=len(training) / GENERATOR_BATCH_SIZE * NUM_AUGMENTS,
        validation_steps=len(validate) / GENERATOR_BATCH_SIZE * NUM_AUGMENTS,
        validation_data=data_generator(validate, GENERATOR_BATCH_SIZE),
        epochs=3)

    model.save(output_path)

def data_generator(samples, entries_per_batch=32):
    """ Generate a data set for a given number of entries. Data is generated
        for each entry and permutation, so this generator will return 
        (entires * permutations) """
    num_samples = len(samples)

    # Loop forever. Keras handles termination of the generator
    while True:
        shuffled = shuffle(samples)

        for offset in range(0, num_samples, entries_per_batch):
            images = []
            angles = []

            batch = shuffled[offset:offset + entries_per_batch]

            for perm in AUGMENTS:
                for sample in batch:
                    img, angle = perm(sample)

                    images.append(img)
                    angles.append(angle)

            yield shuffle(np.array(images), np.array(angles))


def full_path_to_relative(full_path, rel_path):
    return os.path.join(rel_path, 'IMG', os.path.split(full_path)[-1])


def load_data(search_path):
    paths = os.listdir(search_path)
    dirs = []

    for path in paths:
        updated_dir = os.path.join(search_path, path)
        if os.path.isdir(updated_dir):
            dirs.append(updated_dir)

    log_entries = []

    for current_dir in dirs:
        log_path = os.path.join(current_dir, 'driving_log.csv')
        print("Loading data from", log_path)

        if not os.path.exists(log_path):
            print(" Driving log does not exist. Skipping.")
            continue

        entries = []

        with open(log_path, 'r') as log_csv:
            reader = csv.reader(log_csv)
            for line in reader:
                entries.append(
                    de.DrivingEntry(
                        full_path_to_relative(line[0], current_dir),     # center
                        full_path_to_relative(line[1], current_dir),     # left
                        full_path_to_relative(line[2], current_dir),     # right
                        float(line[3]),                     # steering
                        float(line[4]),                     # throttle
                        float(line[5]),                     # braking
                        float(line[6]),                     # speed
                        log_path))

        print(' Loaded {} entries'.format(len(entries)))
        log_entries.extend(entries)

    print('Loaded {} total entries'.format(len(log_entries)))
    return log_entries


if __name__ == '__main__':
    run(build_model)