import math
import random
import numpy as np

# Your manual sigmoid implementation
def manual_sigmoid(x):
    # Activation function for sigmoid is f(x) = 1 / (1 + e^(-x))
    if isinstance(x, (int, float)):
        return 1 / (1 + math.exp(-x))

    if isinstance(x, list):
        return [manual_sigmoid(elem) for elem in x]

# NumPy-based sigmoid function
def numpy_sigmoid(x):
    # Activation function for sigmoid using NumPy
    return 1 / (1 + np.exp(-x))

# Helper function to generate a random n-dimensional list
def generate_random_list(dimensions, low=-5, high=5):
    if len(dimensions) == 1:  # Base case: 1D list
        return [random.uniform(low, high) for _ in range(dimensions[0])]
    else:  # Recursive case: higher dimensions
        return [generate_random_list(dimensions[1:], low, high) for _ in range(dimensions[0])]

# Convert list into NumPy array (recursively for nested lists)
def to_numpy(x):
    if isinstance(x, list):
        return np.array([to_numpy(elem) for elem in x])
    else:
        return x

# Test various dimensions of lists and compare with NumPy
def test_sigmoid():
    test_dimensions = [
        [5],                # 1D list with 5 elements
        [3, 4],             # 2D list (3x4 matrix)
        [2, 3, 4],          # 3D list (2x3x4 tensor)
        [2, 2, 3, 4],       # 4D list (2x2x3x4 tensor)
    ]
    
    for dims in test_dimensions:
        print(f"Testing sigmoid on list of dimensions {dims}...")
        
        # Generate random data and convert to numpy arrays for comparison
        random_list = generate_random_list(dims)
        numpy_array = to_numpy(random_list)
        
        # Compute manually and with NumPy
        manual_result = manual_sigmoid(random_list)
        numpy_result = numpy_sigmoid(numpy_array)
        
        # Compare the results
        print("Manual Sigmoid Result:")
        print(np.array(manual_result))  # Display result in numpy array form for easier comparison
        
        print("NumPy Sigmoid Result:")
        print(numpy_result)
        
        # Check if the manual sigmoid and NumPy sigmoid are approximately equal
        comparison = np.allclose(np.array(manual_result), numpy_result)
        print(f"Is the manual and NumPy result equal? {comparison}")
        
        print("-" * 50)

# Run the test
if __name__ == "__main__":
    test_sigmoid()
