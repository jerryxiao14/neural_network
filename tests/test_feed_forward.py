import numpy as np
import math

# Util function for the sigmoid activation function
def sigmoid(x):
    # Activation function for sigmoid is f(x) = 1/(1+e^(-x))
    if isinstance(x,(int,float)):
        return 1/(1+math.exp(-x))
    
    if isinstance(x,list):
        return [sigmoid(elem) for elem in x]


# Assuming util.dot function is defined
def dot(x,y):
    if len(x[0]) != len(y):
        raise ValueError("Number of columns in matrix a must equal matrix b")
    
    rows_x, cols_x = len(x), len(x[0])
    rows_y, cols_y = len(y), len(y[0])


    result = [[0 for _ in range(cols_y)] for _ in range(rows_x)]
    for i in range(rows_x):
        for j in range(cols_y):
            for k in range(cols_x):
                result[i][j] += x[i][k] * y[k][j]
    return result 

# The Neuron class you've implemented
class Neuron:
    def __init__(self, weights, bias, activation=sigmoid):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def feed_forward(self, inputs):
        total = dot(self.weights, inputs)
        
        if isinstance(total, (int, float)):
            return self.activation(total + self.bias)
        
        elif isinstance(total, list):
            return self.activation(self.apply_bias_elementwise(total, self.bias))

    def apply_bias_elementwise(self, total, bias):
        if isinstance(total, list):
            if isinstance(total[0], (list, tuple)):
                return [self.apply_bias_elementwise(row, bias) for row in total]
            else:
                if isinstance(bias, (list, tuple)):
                    return [total[i] + bias[i] for i in range(len(total))]
                else:
                    return [elem + bias for elem in total]
        raise ValueError("The total must be a list, tuple, or higher-dimensional structure.")

# Utility function for generating a neural network to test comparison
def run_comparison_test():
    # Example Inputs and Weights
    weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    bias = [0.5, 0.7]  # Can be scalar or matching the size of weights

    inputs = [[1, 2,3],[4,5, 6]]

    # Instantiate the custom Neuron object
    custom_neuron = Neuron(weights, bias)

    # Get result from custom neuron
    custom_output = custom_neuron.feed_forward(inputs)

    # Expected output using NumPy
    np_weights = np.array(weights)
    np_inputs = np.array(inputs)
    np_bias = np.array(bias)

    # Compute output in NumPy
    np_total = np.dot(np_weights, np_inputs.T)  # Matrix multiplication (dot product)
    np_result = sigmoid(np_total + np_bias[:, np.newaxis])  # Apply bias

    # Run the test
    print("Custom Neuron output: \n", custom_output)
    print("NumPy output: \n", np_result)

    # Comparing results (tolerances for potential floating point differences)
    np.testing.assert_allclose(custom_output, np_result.tolist(), atol=1e-6)
    print("Test passed, results are close to the expected outputs from NumPy.")

# Call to run the comparison test
run_comparison_test()
