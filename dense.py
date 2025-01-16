
import util
from neuron import Neuron

import random 

class Dense:
    def __init__(self, num_neurons, input_size, activation = util.sigmoid):
        self.num_neurons = num_neurons
        self.input_size = input_size 
        self.activation = activation
        self.initialize_neurons()
    
    def initialize_neurons(self):
        self.neurons = []
        scale = 1
        for _ in range(self.num_neurons):
            weights = [random.uniform(-scale,scale) for _ in range(self.input_size)]
            bias = random.uniform(-scale,scale)
            self.neurons.append(Neuron,(weights,bias, self.activation))

    def forward(self, inputs):
        output = [neuron.feed_forward(inputs) for neuron in self.neurons]
        return output 
    