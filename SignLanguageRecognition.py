import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as p77lt
# import matplotlib.image as mpimg
# from matplotlib import style
# from sklearn import preprocessing, neighbors, svm
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
# import math
# import os
# import cv2
# import random
import pickle
import time


# X = np.load('sign_images_test.npy')
# y = np.load('sign_indexes_test.npy')

# X = np.load('sign_images.npy')
# y = np.load('sign_indexes.npy')

X = np.load('sign_images_128x128_Full_CNN.npy')
y = np.load('sign_indexes_128x128_Full_CNN.npy')

model_name = f"SLR-5Е-CNN-{int(time.time())}"
# tensorboard = TensorBoard(log_dir=f"logs\\{model_name}")

X = X/255.0

X, y = shuffle(X, y)
print(X.shape[1:])

model = Sequential()

model.add(Conv2D(32, 3, 3, input_shape=X.shape[1:]))
model.add(Activation('relu'))

model.add(Dropout(0.5))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(256))
# model.add(Activation('relu'))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))


model.add(Dense(62))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['acc'])


# model.compile(loss=,
#    optimizer='adam',
#    metrics=['acc'])

print(X)
print(y)
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.25)
# model.fit(X, y, batch_size=32, epochs=1, validation_split=0.25, callbacks=[tensorboard])
model.save('SLR-3E-2nd0-128x128-Full-CNN.model')

# print(sign_images_array.shape)
#
# sign_images_array = cv2.resize(sign_images_array, (128, 128))
# plt.imshow(sign_images_array, cmap='gray')
# plt.show()
