Neural Network Implementation
This project is a custom implementation of a neural network from scratch. The goal is to build a fully functional neural network without relying on high-level machine learning frameworks like TensorFlow or PyTorch.

Features
Forward and backward propagation
Customizable activation functions
Training with gradient descent
Support for multiple layers and neurons

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Example usage:

python
Copy
Edit
from neural_network import NeuralNetwork

# Define network architecture
nn = NeuralNetwork(layers=[2, 4, 1], activation="relu")

# Train the model
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Make predictions
predictions = nn.predict(X_test)
To-Do
Add support for additional activation functions
Implement batch normalization
Improve optimization algorithms
Contributing
Feel free to submit issues or pull requests!
