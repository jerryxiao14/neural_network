import util


class Neuron:
    def __init__(self, weights, bias, activation = util.sigmoid):
        self.weights = weights
        self.bias = bias 
        self.activation = activation
    def feed_forward(self, inputs):
        total = util.dot(self.weights, inputs)
        
        # If total is scalar, you directly add bias
        if isinstance(total, (int, float)):
            return self.activation(total + self.bias)
        
        # If total is a list (or higher dimension), we must handle broadcasting and apply bias element-wise
        if isinstance(total, list):
            return self.activation(self.apply_bias_elementwise(total, self.bias))

    def apply_bias_elementwise(self, total, bias):
        # For element-wise addition of bias based on the dimensionality of total and bias
        if isinstance(total, list):
            if isinstance(total[0], (list, tuple)):  # Handle case for 2D or higher
                return [self.apply_bias_elementwise(row, bias) for row in total]
            else:  # Handle 1D case (list of scalars)
                if isinstance(bias, (list, tuple)):  # Bias should be of the same length as total for 1D case
                    return [total[i] + bias[i] for i in range(len(total))]
                else:  # Scalar bias
                    return [elem + bias for elem in total]
        
        # This would handle scalar `total`, but we expect to handle list or higher only.
        raise ValueError("The total should be a list, tuple, or higher dimensional structure.")
            # assume it is a list here
            
            

    