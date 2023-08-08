import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# Load CSV files for weights and biases
weights_layer_1 = np.loadtxt('layer_1_weights.csv', delimiter=',')
biases_layer_1 = np.loadtxt('layer_1_biases.csv', delimiter=',')

weights_layer_2 = np.loadtxt('layer_2_weights.csv', delimiter=',')
biases_layer_2 = np.loadtxt('layer_2_biases.csv', delimiter=',')

weights_layer_3 = np.loadtxt('layer_3_weights.csv', delimiter=',')
biases_layer_3 = np.loadtxt('layer_3_biases.csv', delimiter=',')

# Create the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu', weights=[weights_layer_1, biases_layer_1]),
    Dropout(0.3),
    Dense(512, activation='relu', weights=[weights_layer_2, biases_layer_2]),
    Dropout(0.3),
    Dense(512, activation='relu', weights=[weights_layer_3, biases_layer_3]),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the model (you may need to set optimizer, loss, and metrics)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
