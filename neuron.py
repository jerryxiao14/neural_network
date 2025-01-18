import util


class Neuron:
    def __init__(self, weights, bias, activation = util.sigmoid, activation_grad = util.sigmoid_derivative):
        self.weights = weights
        self.bias = bias 
        self.activation = activation
        self.activation_grad = activation_grad
    def feed_forward(self, inputs):
        self.input_cache = inputs
        total = util.dot(self.weights, inputs)
        # If total is scalar, you directly add bias
        if isinstance(total, (int, float)):
            self.output = self.activation(total + self.bias)
            return self.output
        
        # If total is a list (or higher dimension), we must handle broadcasting and apply bias element-wise
        if isinstance(total, list):
            self.output = self.activation(self.apply_bias_elementwise(total,self.bias))
            return self.output

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
    
    def multiply_element_wise(self, a,b):
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            return a*b
        return [self.multiply_element_wise(a[i],b[i]) for i in range(len(a))]
    def backward(self, grad):
        # Compute gradient with respect to weights and bias
        grad_activation = self.activation_grad(self.output)

        grad_total = self.multiply_element_wise(grad, grad_activation)

        self.grad_weights = [grad_total * inp for inp in self.input_cache]
        self.grad_bias = grad_total 

        grad_inputs = [grad_total * w for w in self.weights]

        return grad_inputs
    
    def update_params(self, learning_rate = 0.01):
        self.weights = [w - learning_rate * grad_w for w, grad_w in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias 

        
        
    