# https://youtu.be/PGI84VwTT-4?si=PIwiPXqJ_cZlP5Je

import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization
from keras.models import Sequential, Model,
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()


np.random.seed(777)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'sheep', 'truck']

sample_size = 9
random_idx = np.random.randint(60000,size=sample_size)

# plt.figure(figsize=(5,5))
# for i, idx in enumerate(random_idx):
#     plt.subplot(3, 3, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train_full[i])
#     plt.xlabel(class_names[int(y_train_full[i])])

plt.show()

x_mean = np.mean(x_train_full, axis=(0, 1, 2))
x_std = np.std(x_train_full, axis=(0, 1, 2))

x_train_full = (x_train_full - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=.3)

# print(x_train.shape)
# print(x_train.shape)
#
# print(x_test.shape)
# print(y_test.shape)
#
# print(x_val.shape)
# print(y_val.shape)

def model_build():
    model = Sequential()

    input = Input(shape=(32, 32, 3))

    output = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(input)
    output = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(output)

    output = Flatten()(output)
    output = Dense(256, activation='relu')(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='softmax')(output)

    model = Model(inputs=[input], outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='spars-catrgorical_crossentropy', metrics=['accuracy'])

    return model
model = model_build()
model.summary()