import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from neuron import Neuron
from nn import NeuralNetwork
from dense import Dense
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from nn import NeuralNetwork
import util
import loss


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten and normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0     # Flatten and normalize
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
    X_train, y_train_encoded, test_size=0.1, random_state=42
)

nn = NeuralNetwork()
layer_one = Dense(num_neurons=128, input_size=784)
layer_two = Dense(num_neurons=64, input_size=128)
layer_three = Dense(num_neurons=10, input_size=64)

nn.add_layer(layer_one)  # First hidden layer with sigmoid
nn.add_layer(layer_two)   # Second hidden layer with sigmoid
nn.add_layer(layer_three)    # Output layer with softmax


# Lets add a line of comment here 

nn.train(
    inputs = X_train, 
    targets = y_train_encoded, 
    loss_function = loss.cross_entropy_loss, 
    loss_grad_function= loss.cross_entropy_loss_derivative,
    epochs = 100,
    learning_rate = 0.01 
)