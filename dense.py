
import util
from neuron import Neuron

import random 

class Dense:
    def __init__(self, num_neurons, input_size, activation = util.sigmoid, activation_grad = util.sigmoid_derivative):
        self.num_neurons = num_neurons
        self.input_size = input_size 
        self.activation = activation
        self.activation_grad = activation_grad
        self.initialize_neurons()
    
    def initialize_neurons(self):
        self.neurons = []
        scale = 1
        for _ in range(self.num_neurons):
            weights = [random.uniform(-scale,scale) for _ in range(self.input_size)]
            bias = random.uniform(-scale,scale)
            neuron = Neuron(weights, bias, self.activation, self.activation_grad)
            self.neurons.append(neuron)

    def forward(self, inputs):
        output = [neuron.feed_forward(inputs) for neuron in self.neurons]
        return output 
    
    def backward(self, grads):
        grad_inputs = [0 for _ in range(self.input_size)]
        for i, neuron in enumerate(self.neurons):
            grad_input = neuron.backward(grads[i])
            grad_inputs = [grad_inputs[j] + grad_input[j] for j in range(self.input_size)]
        return grad_inputs

    def update_params(self, learning_rate = 0.01):
        for neuron in self.neurons:
            neuron.update_params(learning_rate)
    