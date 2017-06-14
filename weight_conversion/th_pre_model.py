import os
os.environ["KERAS_BACKEND"] = "device"
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D


def theano_model():

    model = Sequential()
    model.add(Convolution2D(32, 8, strides=(4,4), input_shape=(4, 84, 84), data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, strides=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=(1,1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(6))
    return model
