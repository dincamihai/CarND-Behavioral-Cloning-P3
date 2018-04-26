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
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad
from keras.models import load_model
from keras.regularizers import l2
from sklearn.utils import shuffle
from skimage.color import rgb2hsv
from matplotlib import pyplot as plt


MODEL_FILE = 'model.h5'



from keras.backend import tf as ktf
def create_model(load=False):
    inputs = Input(shape=(160, 320, 3))
    main = Cropping2D(cropping=((75,20), (0,0)), input_shape=(160, 320, 3))(inputs)
    main = Lambda(lambda x: x/255.0 - 0.5)(main)
    main = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), border_mode='valid')(main)

    main = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid')(main)
    main = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(main)
    main = Activation('relu')(main)
    main = BatchNormalization(
        epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(main)
    main = Dropout(0.5)(main)

    main = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid')(main)
    main = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(main)
    main = Activation('relu')(main)
    main = BatchNormalization(
        epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(main)
    main = Dropout(0.5)(main)

    main = Flatten()(main)

    main = Dense(50, W_regularizer=l2(0.01))(main)
    main = Activation('relu')(main)

    main = Dense(20, W_regularizer=l2(0.01))(main)
    main = Activation('relu')(main)

    main = Dense(1)(main)
    model = Model(input=inputs, output=main)
    if load:
        model.load_weights(MODEL_FILE)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


def main():
    correction = 0.9

    def fun(path, usehsv=False):
        assert path
        image = cv2.imread('/'.join(['.'] + path.split('/')[-4:]))
        if not usehsv:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([0,0,0])
        upper_blue = np.array([60,100,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imwrite('image.png', hsv)
        # cv2.imwrite('mask.png', mask)
        # cv2.imwrite('out.png', res)
        # assert image.shape == (160, 320, 3)
        return res 

    images = []
    angles = []
    paths = [
        # 'data/bridge',
        # 'data/curve-correction',
        # 'data/curves-track2',
        # 'data/edge-to-center',
        # 'data/forward-center-track2',
        # 'data/forward-left-track2',
        # 'data/forward-slow',
        # 'data/forward-track2',
        # 'data/forward',
        # 'data/more-recovery-track2',
        # 'data/recovery-track2',
        # 'data/reverse-left-track2',
        # 'data/reverse-slow',
        # 'data/reverse-track2',
        # 'data/reverse',
        # 'data/track2',
        # 'data/unclear-edge',
        'data.v1/track1-forward',
        'data.v1/track1-recovery',
        'data.v1/track1-reverse',
        # 'data.v1/track2-forward',
        'data.v1/track2-recovery'
    ]

    for item in paths:
        print("Processing %s" % item)
        for chunk in pandas.read_csv(
            '/'.join(['.', item, 'driving_log.csv']),
            names=['center', 'left', 'right', 'angles', 'x', 'y', 'speed'],
            chunksize=3000
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
