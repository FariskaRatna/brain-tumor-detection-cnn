import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import keras
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from tensorflow.keras import backend as K
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from tqdm import tqdm
import cv2
import mplcyberpunk
plt.style.use("cyberpunk")

from cnn_model import cnn_model

X_train = []
y_train = []
labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]

image_size=512
for i in labels:
    folder_path = os.path.join('/kaggle/input/brain-tumor-classification-mri/Training', i)
    for j in tqdm(os.listdir(folder_path)):
        img=cv2.imread(os.path.join(folder_path,j))
        img=cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
for i in labels:
    folder_path = os.path.join('/kaggle/input/brain-tumor-classification-mri/Testing', i)
    for j in tqdm(os.listdir(folder_path)):
        img=cv2.imread(os.path.join(folder_path, j))
        img=cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)
        
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

model = cnn_model()

checkpoint_path='best_23cnn.weights.h5'
checkpoint_callback=ModelCheckpoint(filepath=checkpoint_path,
                                   monitor='val_loss',
                                   save_best_only=True,
                                   save_weights_only=True,
                                   verbose=1)
early_stop=EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, verbose=1, batch_size=32, callbacks=[checkpoint_callback, early_stop])
model.load_weights("best_23cnn.weights.h5")