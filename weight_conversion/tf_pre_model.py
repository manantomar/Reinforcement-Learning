from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D

class tensorflow_model:
    def __init__(self, ):

        model = Sequential()
        model.add(Convolution2D(32, 8, strides=(4,4), input_shape=(84, 84, 4), data_format="channels_last"))
        #model.layers[1].set_weights(param_values["w1"])
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, strides=(2,2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, strides=(1,1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(6))
