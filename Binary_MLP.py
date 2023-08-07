import numpy as np
import tensorflow as tf
from keras.optimizers import Nadam, Ftrl, Adagrad
from keras.losses import mean_squared_error, sparse_categorical_crossentropy, hinge, kullback_leibler_divergence

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

best_accuracy = 0.0
best_batch_size = 0
best_validation_split = 0.0
lose = None
best_weights = None

batch_sizes = [32, 64, 128, 256, 512]  # 다양한 배치 크기 시도
validation_splits = [0.1, 0.15, 0.2, 0.25, 0.3]  # 다양한 밸리데이션 스플릿 시도
optimizers = ['Nadam', 'FTrl', 'Adagrad']
loss_functions = ['mean_squared_error', 'sparse_categorical_crossentropy', 'hinge', 'kullback_leibler_divergence']

'''===============================================================================
activation = Nadam
optimizer = Adamax
loss = mean_squared_error
==============================================================================='''

for batch_size in batch_sizes:
    for validation_split in validation_splits:
        for optimizer in optimizers:
            for loss in loss_functions:
                print("="*50)
                print("Batch Size:", batch_size)
                print("Validation Split:", validation_split)
                print("Optimizer:", optimizer)
                print("Loss Function:", loss)
                print("="*50)

                model = Sequential([
                  Flatten(input_shape=(28, 28)),
                  Dense(512, activation='sigmoid'),
                  Dropout(0.5),
                  Dense(512, activation='sigmoid'),
                  Dropout(0.5),
                  Dense(512, activation='sigmoid'),
                  Dropout(0.5),
                  Dense(10, activation='sigmoid')  # 출력 뉴런 개수를 10으로 설정
                  ])
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=['accuracy']
                             )

                early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

                model.fit(train_x, train_y, epochs=100, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

                loss, accuracy = model.evaluate(test_x, test_y)
                print("Batch Size:", batch_size, " Validation Split:", validation_split, " Loss Function:", mean_squared_error)
                print("Accuracy:", accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_batch_size = batch_size
                    best_validation_split = validation_split
                    best_weights = model.get_weights()

# 가장 높은 정확도의 가중치를 모델에 설정
model.set_weights(best_weights)
loss, accuracy = model.evaluate(test_x, test_y)
print("Best Test Accuracy:", accuracy)
print("Best Batch Size:", best_batch_size)
print("Best Validation Split:", best_validation_split)


