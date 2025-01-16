import util
from neuron import Neuron
import random

class Conv2D:
    def __init__(self, num_filters, filter_size, input_shape, stride = 1, padding = 'valid', activation_function = None):
        self.num_filters = num_filters
        self.filter_size = filter_size 
        self.input_shape = input_shape
        self.stride = stride 
        self.padding = padding 
        self.activation_function = None 

        self.initialize_filters()
        self.biases = [random.random() for _ in range(self.num_filters)]
    
    def initialize_filters(self):
        self.filters = []
        for _ in range(self.num_filters):
            filter_mat = []
            for _ in range(self.filter_size[0]):
                filter_mat.append([[random.random() for _ in range(self.input_shape[2])] for _ in range(self.filter_size[1])])
            self.filters.append(filter_mat)
    
    def apply_padding(self,inputs):
        if self.padding == 'same':
            pad_height = (self.filter_size[0]-1)//2 
            pad_width = (self.filter_size[1]-1)//2
            padded_input = []

            for channel in range(len(inputs[0][0])):
                padded_channel =[[0] * (len(inputs[0]) + 2*pad_width)
                                 for _ in range(len(inputs)+2*pad_height)]
                for i in range(len(inputs)):
                    for j in range(len(inputs[0])):
                        padded_channel[i+pad_height][j+pad_width]=inputs[i][j][channel]
                    padded_input.append(padded_channel)
            return padded_input 
        return inputs

    def element_wise_product_sum(self, slice_input, filter, bias):
        total = 0
        for i in range(len(filter)):
            for j in range(len(filter[0])):
                for k in range(len(filter[0][0])):
                    total += slice_input[i][j][k]*filter[i][j][k]
        return total+bias
    
    def convolve(self,inputs):
        inputs = self.apply_padding(inputs)

        input_height = len(inputs)
        input_width = len(inputs[0])
        input_depth = len(inputs[0][0])

        output_height = (input_height-self.filter_size[0])//self.stride + 1
        output_width = (input_width - self.filter_size[1])//self.stride + 1

        output = [[[0.0 for _ in range(self.num_filters)] for _ in range(output_width)] for _ in range(output_height)]

        for i in range(output_height):
            for j in range(output_width):
                for f in range(self.num_filters):
                    vertical_start = i*self.stride 
                    vertical_end = vertical_start + self.filter_size[0]
                    horizontal_start = j*self.stride 
                    horizontal_end = horizontal_start + self.filter_size[1]
                
                    slice_input = [
                        inputs[vertical_start+x][horizontal_start:horizontal_end]
                        for x in range(self.filter_size[0])
                    ]

                    value = self.element_wise_product_sum(slice_input, self.filters[f], self.biases[f])
                    
                    if self.activation_function:
                        value = self.activation_function(value)
                output[i][j][f]=value
        return output