import os
import datetime
import pickle
import pandas
import numpy as np
import cv2
import h5py
import glob
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Convolution2D, Activation, Lambda, Dropout, merge
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Cropping2D
from keras.models import load_model
from keras.regularizers import l2
from sklearn.utils import shuffle


MODEL_FILE = 'model.h5'
# ALLOWED_PATTERN = 'data/*[!-track2]'
ALLOWED_PATTERN = 'data.v1/*'


def create_model(load=False):
    inputs = Input(shape=(160, 320, 3))
    main = Cropping2D(cropping=((75,20), (0,0)), input_shape=(160, 320, 3))(inputs)
    main = Lambda(lambda x: x/255.0 - 0.5)(main)
    main = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(main)

    right = Convolution2D(24, 5, 5, border_mode='valid')(main)
    right = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(right)
    right = Activation('relu')(right)
    right = Dropout(0.3)(right)

    right = Convolution2D(48, 3, 3, border_mode='valid')(main)
    right = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(right)
    right = Activation('relu')(right)
    right = Dropout(0.5)(right)

    right = Convolution2D(48, 2, 2, border_mode='valid')(right)
    right = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(right)
    right = Activation('relu')(right)
    right = Dropout(0.5)(right)

    left = Convolution2D(48, 5, 5, border_mode='valid')(main)
    left = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(left)
    left = Activation('relu')(left)
    left = Dropout(0.5)(left)

    left = Convolution2D(48, 2, 2, border_mode='valid')(left)
    left = AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(left)
    left = Activation('relu')(left)
    left = Dropout(0.5)(left)

    main = merge([Flatten()(left), Flatten()(right)], mode='concat')

    main = Dense(50, W_regularizer=l2(0.009))(main)
    main = Activation('relu')(main)

    main = Dense(20, W_regularizer=l2(0.009))(main)
    main = Activation('relu')(main)

    main = Dense(1)(main)
    model = Model(input=inputs, output=main)
    if load:
        model.load_weights(MODEL_FILE)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


def main():
    correction = 0.4

    def fun(path):
        assert path
        image = cv2.imread('/'.join(['.'] + path.split('/')[-4:]))
        # assert image.shape == (160, 320, 3)
        return image

    images = []
    angles = []
    for item in glob.glob(ALLOWED_PATTERN):
        print("Processing %s" % item)
        for chunk in pandas.read_csv(
            '/'.join(['.', item, 'driving_log.csv']),
            names=['center', 'left', 'right', 'angles', 'x', 'y', 'speed'],
            chunksize=1000
        ):
            center_images = chunk.loc[:, 'center']
            center_angles = chunk.loc[:, 'angles']

            left_images = chunk.loc[:, 'left']
            left_angles = correction + center_angles

            right_images = chunk.loc[:, 'right']
            right_angles = -correction + center_angles

            images += center_images.tolist()
            angles += center_angles.tolist()

            images += left_images.tolist()
            angles += left_angles.tolist()

            images += right_images.tolist()
            angles += right_angles.tolist()

    from sklearn.model_selection import train_test_split
    train_images, valid_images, train_angles, valid_angles = train_test_split(images, angles, test_size=0.2, random_state=13)

    def generator(images, angles, batch_size=32):
        num_samples = len(images)
        while 1: # Loop forever so the generator never terminates
            shuffle(images, angles)
            for offset in range(0, num_samples, batch_size):
                batch_images = images[offset:offset+batch_size]
                batch_angles = angles[offset:offset+batch_size]

                gen_images = []
                gen_angles = []
                for path, angl in zip(batch_images, batch_angles):
                    image = fun(path)
                    if image is None:
                        print("Skipping image: %s" % path)
                        continue
                    gen_images.append(image)
                    gen_angles.append(angl)
                    gen_images.append(np.fliplr(image))
                    gen_angles.append(-angl)

                # trim image to only see section with road
                X_train = np.array(gen_images)
                y_train = np.array(gen_angles)
                yield shuffle(X_train, y_train)

    train_generator = generator(train_images, train_angles, batch_size=128)
    valid_generator = generator(valid_images, valid_angles, batch_size=128)

    model = create_model()

    num_train_images = (2 * len(train_images) / 128)
    samples_per_epoch = num_train_images * 128

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=samples_per_epoch,
        validation_data=valid_generator,
        nb_val_samples=samples_per_epoch*0.2,
        nb_epoch=3,
        verbose=1)

    model_file = 'model.h5-{0}-{1}'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"), correction)
    model.save(model_file)
    print("%s saved" % model_file)

    with open('history.p', 'wb') as _file:
        pickle.dump(history.history, _file)


if __name__ == "__main__":
    main()
