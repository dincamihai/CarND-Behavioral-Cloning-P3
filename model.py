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


def create_model():
    try:
        raise
        return load_model(MODEL_FILE)
    except:
        model = Sequential()
        model.add(Cropping2D(cropping=((75,20), (0,0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: x/255 - 0.5))
        model.add(Convolution2D(16, 5, 5, border_mode='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='tf'))
        model.add(Activation('relu'))
        # model.add(
        #     BatchNormalization(
        #         epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)
        # )

        model.add(Dropout(0.5))
        model.add(Convolution2D(64, 5, 5, border_mode='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        model.add(Activation('relu'))

        # model.add(
        #     BatchNormalization(
        #         epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)
        # )
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(10, W_regularizer=l2(0.1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(10, W_regularizer=l2(0.1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        return model


def main():
    images = []
    angles = []

    def fun(path):
        try:
            assert path
        except:
            import pdb; pdb.set_trace()
        image = cv2.imread('./{0}/IMG/'.format(DATA_FOLDER) + path.split('/')[-1])
        assert image.shape == (160, 320, 3)
        return image

    for chunk in pandas.read_csv(
        './%s/driving_log.csv' % DATA_FOLDER,
        names=['center', 'left', 'right', 'angles', 'x', 'y', 'speed'],
        chunksize=1000
    ):
        center_images_chunk = chunk.loc[:, 'center']
        left_images_chunk = chunk.loc[:, 'left']
        right_images_chunk = chunk.loc[:, 'right']

        images += center_images_chunk.tolist()
        angles += chunk.loc[:, 'angles'].tolist()

        images += left_images_chunk.tolist()
        angles += (0.2 + chunk.loc[:, 'angles']).tolist()

        images += right_images_chunk.tolist()
        angles += (-0.2 + chunk.loc[:, 'angles']).tolist()

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
                    gen_images.append(image)
                    gen_angles.append(angl)
                    gen_images.append(np.fliplr(image))
                    gen_angles.append(-angl)

                # trim image to only see section with road
                X_train = np.array(gen_images)
                y_train = np.array(gen_angles)
                yield shuffle(X_train, y_train)

    train_generator = generator(train_images, train_angles, batch_size=32)
    valid_generator = generator(valid_images, valid_angles, batch_size=32)

    model = create_model()
    model.fit_generator(
        train_generator,
        samples_per_epoch=3 * 2 * len(train_images),
        validation_data=valid_generator,
        nb_val_samples=len(valid_images),
        nb_epoch=3)
    model.save('model.h5')


if __name__ == "__main__":
    main()
