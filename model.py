import os
import pickle
import pandas
import numpy as np
import cv2
import h5py
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Activation, Lambda, Dropout
from keras.layers.pooling import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Cropping2D
from keras.models import load_model
from keras.regularizers import l2
from sklearn.utils import shuffle


MODEL_FILE = 'model.h5'
DATA_FOLDER = './data'


def create_model(load=False):
    if load:
        return load_model(MODEL_FILE)
    else:
        model = Sequential()
        model.add(Cropping2D(cropping=((75,20), (0,0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: x/255.0 - 0.5))

        model.add(Convolution2D(24, 5, 5, border_mode='valid'))
        model.add(AveragePooling2D(pool_size=(1, 1), strides=None, border_mode='valid', dim_ordering='tf'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(36, 5, 5, border_mode='valid'))
        model.add(AveragePooling2D(border_mode='valid', dim_ordering='default'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(48, 5, 5, border_mode='valid'))
        model.add(AveragePooling2D(border_mode='valid', dim_ordering='default'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64, 3, 3, border_mode='valid'))
        model.add(AveragePooling2D(pool_size=(1, 1), border_mode='valid', dim_ordering='default'))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, 3, 3, border_mode='valid'))
        model.add(AveragePooling2D(pool_size=(1, 1), border_mode='valid', dim_ordering='default'))
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(1164, W_regularizer=l2(0.1)))
        model.add(Activation('relu'))

        model.add(Dense(100, W_regularizer=l2(0.1)))
        model.add(Activation('relu'))

        model.add(Dense(50, W_regularizer=l2(0.1)))
        model.add(Activation('relu'))

        model.add(Dense(10, W_regularizer=l2(0.1)))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model


def main():

    def fun(path):
        assert path
        image = cv2.imread('/'.join(['.'] + path.split('/')[-4:]))
        # assert image.shape == (160, 320, 3)
        return image

    images = []
    angles = []
    data_root = './%s' % DATA_FOLDER
    for item in os.listdir(data_root):
        for chunk in pandas.read_csv(
            '/'.join([data_root, item, 'driving_log.csv']),
            names=['center', 'left', 'right', 'angles', 'x', 'y', 'speed'],
            chunksize=1000
        ):
            images += chunk.loc[:, 'center'].tolist()
            angles += chunk.loc[:, 'angles'].tolist()

            correction = 0.5
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

    train_generator = generator(train_images, train_angles, batch_size=32, augment=True)
    valid_generator = generator(valid_images, valid_angles, batch_size=32)

    model = create_model()
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=32*1000,
        validation_data=valid_generator,
        nb_val_samples=len(valid_images),
        nb_epoch=2,
        verbose=1)

    model.save('model.h5')

    with open('history.p', 'wb') as _file:
        pickle.dump(history.history, _file)


if __name__ == "__main__":
    main()
