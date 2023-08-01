# # https://youtu.be/WvoLTXIjBYU?si=8qoc5Jg8DoTUVXDR
# # Convolutional Neural Networks - Python, TensorFlow 및 Keras를 사용한 딥 러닝 기본 사항 p.3
#
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# import pickle
#
# X = pickle.load(open("X.pickle", "rb"))
# y = pickle.load(open("y.pickle", "rb"))
#
# X = X/255.0
#
# # First Layer
# model = Sequential()
# model.add(Conv2D((64), (3, 3), input_shape=X.shape[1:]))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# # Second Layer
# model.add(Conv2D((64), (3, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# # Third Layer
# model.add(Flatten())
# model.add(Dense(64))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss="binary_crossentropy",
#               optimizer="adam",
#               metrics=['accuracy'])
#
# model.fit(X, y,batch_size=32, validation_split=0.1)

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(train_x, train_y, epochs=100, batch_size=128, validation_split=0.1, callbacks=[early_stopping])

loss, accuracy = model.evaluate(test_x, test_y)

# 가장 높은 정확도의 가중치 추출
best_weights = model.get_weights()

# 가장 높은 정확도의 가중치를 모델에 설정
model.set_weights(best_weights)
loss, accuracy = model.evaluate(test_x, test_y)
print("Test accuracy:", accuracy)
print("Best weights shape : ", len(best_weights))
print("Best weights : ", best_weights)

print(best_weights[0].shape)
print(best_weights[1].shape)
print(best_weights[2].shape)
print(best_weights[3].shape)
print(best_weights[4].shape)
print(best_weights[5].shape)
print(best_weights[6].shape)
print(best_weights[7].shape)

np.savetxt('weights[0].csv', best_weights[0])
np.savetxt('weights[1].csv', best_weights[1])
np.savetxt('weights[2].csv', best_weights[2])
np.savetxt('weights[3].csv', best_weights[3])
np.savetxt('weights[4].csv', best_weights[4])
np.savetxt('weights[5].csv', best_weights[5])
np.savetxt('weights[6].csv', best_weights[6])
np.savetxt('weights[7].csv', best_weights[7])