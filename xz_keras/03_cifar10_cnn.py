# train a simple deep cnn on the cifar10 small images dataset

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, losses, metrics
import os


class Model():
    def __init__(self):
        self.batch_size = 32
        self.num_classes = 10
        self.epochs = 100
        self.data_augmentation = True
        self.save_dir = os.path.join(os.getcwd(), 'saved_models')
        self.model_name = 'keras_cifar10_trained_model.h5'
        self.model = Sequential()

    def get_data(self):
        # load the data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        # convert to one hot
        self.y_train = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, num_classes=self.num_classes)
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

    def inference(self):
        self.model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.x_train.shape[1:], activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def compile(self):
        self.model.compile(optimizer=optimizers.Adam(),
                           loss=losses.categorical_crossentropy,
                           metrics=[metrics.categorical_accuracy])

    def data_aug(self):
        if not self.data_augmentation:
            print('Not using data augmentation.')
        else:
            print('Using real-time data augmentation.')
            # this will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False
            )
            datagen.fit(self.x_train)
            self.data_aug_ = datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)

    def train_op(self):
        if self.data_augmentation:
            self.model.fit_generator(self.data_aug_, steps_per_epoch=len(self.x_train)/self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=(self.x_test, self.y_test))
        else:
            self.model.fit(self.x_train, self.y_train,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data=(self.x_test, self.y_test),
                           shuffle=True)

    def save_op(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        model_path = os.path.join(self.save_dir, self.model_name)
        self.model.save(model_path)
        print('Saved trained model as {0}'.format(model_path))

    def evaluate_op(self):
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test loss: {0}'.format(scores[0]))
        print('Test acc: {0}'.format(scores[1]))

    def build(self):
        self.get_data()
        self.inference()
        self.compile()
        self.data_aug()
        self.train_op()
        self.save_op()
        self.evaluate_op()


if __name__ == '__main__':
    model = Model()
    model.build()