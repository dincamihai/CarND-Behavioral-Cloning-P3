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
ALLOWED_PATTERN = 'data/*'


def create_model(load=False):
    if load:
        return load_model(MODEL_FILE)
    else:
        inputs = Input(shape=(160, 320, 3))
        main = Cropping2D(cropping=((75,20), (0,0)), input_shape=(160, 320, 3))(inputs)
        main = Lambda(lambda x: x/255.0 - 0.5)(main)
        main = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='tf')(main)

        main = Convolution2D(24, 2, 2, border_mode='valid')(main)
        main = AveragePooling2D(pool_size=(3, 3), strides=None, border_mode='valid', dim_ordering='tf')(main)
        main = Activation('relu')(main)
        main = Dropout(0.5)(main)

        main = Convolution2D(48, 2, 2, border_mode='valid')(main)
        main = AveragePooling2D(pool_size=(3, 3), strides=None, border_mode='valid', dim_ordering='default')(main)
        main = Activation('relu')(main)
        main = Dropout(0.5)(main)

        main = Flatten()(main)

        main = Dense(50, W_regularizer=l2(0.7))(main)
        main = Activation('relu')(main)

        main = Dense(7, W_regularizer=l2(0.5))(main)
        main = Activation('relu')(main)

        main = Dense(1)(main)
        model = Model(input=inputs, output=main)
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model


def main():
    correction = 0.6

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
            images += chunk.loc[:, 'center'].tolist()
            angles += chunk.loc[:, 'angles'].tolist()

            left_images_chunk = chunk.loc[:, 'left']
            right_images_chunk = chunk.loc[:, 'right']

            images += left_images_chunk.tolist()
            angles += (correction + chunk.loc[:, 'angles']).tolist()

            images += right_images_chunk.tolist()
            angles += (-correction + chunk.loc[:, 'angles']).tolist()

    from sklearn.model_selection import train_test_split
    train_images, valid_images, train_angles, valid_angles = train_test_split(images, angles, test_size=0.2, random_state=13)

    def generator(images, angles, batch_size=32, augment=False):
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
                        print("Skipping image: %s" % image)
                        continue
                    if (not augment) and not ('center' in path):
                        continue
                    gen_images.append(image)
                    gen_angles.append(angl)
                    if augment:
                        gen_images.append(np.fliplr(image))
                        gen_angles.append(-angl)

                # trim image to only see section with road
                X_train = np.array(gen_images)
                y_train = np.array(gen_angles)
                yield shuffle(X_train, y_train)

    train_generator = generator(train_images, train_angles, batch_size=128, augment=True)
    valid_generator = generator(valid_images, valid_angles, batch_size=128, augment=True)

    model = create_model()
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_images)/10,
        validation_data=valid_generator,
        nb_val_samples=len(valid_images)/10,
        nb_epoch=5,
        verbose=1)

    model_file = 'model.h5-{0}-{1}'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"), correction)
    model.save(model_file)
    print("%s saved" % model_file)

    with open('history.p', 'wb') as _file:
        pickle.dump(history.history, _file)


if __name__ == "__main__":
    main()
