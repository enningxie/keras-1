# trains a simple deep nn on the mnist dataset

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras import losses, optimizers, metrics

batch_size = 128
num_classes = 10
epochs = 20

# load mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class to one_hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# 1. after handle the dataset, let's define the model.
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# print the structure of your model
model.summary()

# 2. compile your model.
model.compile(optimizer=Adam(lr=0.001),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

# 3. train your model.
his = model.fit(x_train, y_train,
          batch_size=batch_size,
          verbose=1,
          validation_data=(x_test, y_test),
          epochs=epochs)

# 4. evalue your model.
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score[0])
print('Test acc: ', score[1])

