from conv2d import Conv2D
from dense import Dense


class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs 
        for layer in self.layers:
            output = layer.convolve(output) if isinstance(layer, Conv2D) else layer.forward(output)
        return output

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_params(self, learning_rate = 0.01):
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self, inputs, targets, loss_function, loss_grad_function, epochs, learning_rate = 0.01,):
        for epoch in range(epochs):
            print(f'epoch {epoch+1}/{epochs}')
            loss = 0
            for i, input in enumerate(inputs):
                target = targets[i]
                prediction = self.forward(input)
                loss += loss_function(target, prediction)
                loss_grad = loss_grad_function(target, prediction)
                self.backward(loss_grad)
            print(f'epoch {epoch+1}/{epochs}, loss: {loss/len(inputs)}')