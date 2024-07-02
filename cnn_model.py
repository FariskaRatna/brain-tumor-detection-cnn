import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

def cnn_model():
    input_size = keras.Input(shape=(512, 512, 3), name='image')

    x1=layers.Conv2D(64,(22,22),strides=2)(input_size)
    x1=layers.MaxPooling2D((4, 4))(x1)
    x1=layers.BatchNormalization()(x1)

    x2=layers.Conv2D(128,(11,11), strides=2, padding='same')(x1)
    x2=layers.MaxPooling2D((2,2))(x2)
    x2=layers.BatchNormalization()(x2)

    x3=layers.Conv2D(256,(7,7),strides=2,padding='same')(x2)
    x3=layers.MaxPooling2D((2,2))(x3)
    x3=layers.BatchNormalization()(x3)

    x4=layers.Conv2D(512,(2,2),strides=2,padding='same')(x3)
    x4=layers.MaxPooling2D((2,2))(x4)
    x4=layers.BatchNormalization()(x4)

    x5=layers.GlobalAveragePooling2D()(x4)
    x5=layers.Activation("relu")(x5)

    x6=layers.Dense(1024,"relu")(x5)
    x6=layers.BatchNormalization()(x6)
    x7=layers.Dense(512,"relu")(x6)
    x7=layers.BatchNormalization()(x7)
    x8=layers.Dense(256,"relu")(x7)
    x8=layers.BatchNormalization()(x8)
    x8=layers.Dropout(0.2)(x8)
    x9=layers.Flatten()(x8)
    x10=layers.Dense(4)(x9)
    pred=layers.Activation("softmax")(x10)

    model=keras.Model(inputs=input_size, outputs=pred)

    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# model = cnn_model()
# model.summary()