import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from neuron import Neuron
class testNetwork:
    def __init__(self):
        weights = [0,1]
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights,bias)
        self.o1 = Neuron(weights,bias)
    def feedforward(self, x):
        out_h1  = self.h1.feed_forward(x)
        out_h2 = self.h2.feed_forward(x)
        
        out_o1 = self.o1.feed_forward([out_h1,out_h2])

        return out_o1 


test = testNetwork()
x = [2,3]
print(test.feedforward(x))
