
import sys
import os
# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)


import util
import numpy as np
import random
# Define your custom dot function
def dot(x, y):
    if len(x[0]) != len(y):
        raise ValueError("Number of columns in matrix x must equal number of rows in matrix y.")
    
    rows_x, cols_x = len(x), len(x[0])
    rows_y, cols_y = len(y), len(y[0])

    result = [[0 for _ in range(cols_y)] for _ in range(rows_x)]
    for i in range(rows_x):
        for j in range(cols_y):
            for k in range(cols_x):
                result[i][j] += x[i][k] * y[k][j]
    return result

# Test function to compare your dot product with numpy's dot product
def test_dot_function():
    for _ in range(10):  # Test 10 times with different random shapes
        # 1D x 1D test
        size_1d = random.randint(2, 5)  # Random vector size between 2 and 5
        x_1d = [random.randint(1, 10) for _ in range(size_1d)]
        y_1d = [random.randint(1, 10) for _ in range(size_1d)]
        
        custom_result_1d = np.dot(x_1d, y_1d)
        numpy_result_1d = np.dot(x_1d, y_1d)
        
        # Compare for 1D vectors (inner product)
        assert custom_result_1d == numpy_result_1d, f"Failed for 1D vectors of size {size_1d}"
        
        # 2D x 2D test (matrix multiplication)
        rows_x = random.randint(2, 5)
        cols_x = random.randint(2, 5)
        rows_y = cols_x  # columns in x should match rows in y
        cols_y = random.randint(2, 5)
        
        x_2d = [[random.randint(1, 10) for _ in range(cols_x)] for _ in range(rows_x)]
        y_2d = [[random.randint(1, 10) for _ in range(cols_y)] for _ in range(rows_y)]
        
        custom_result_2d = dot(x_2d, y_2d)
        numpy_result_2d = np.dot(x_2d, y_2d)
        
        # Compare for 2D matrices (dot product for matrices)
        assert np.allclose(custom_result_2d, numpy_result_2d.tolist()), f"Failed for 2D matrices of size {rows_x}x{cols_x} and {rows_y}x{cols_y}"
        
        # 3D x 3D test (batched dot product)
        batch_size = random.randint(2, 5)
        depth = random.randint(2, 5)  # Depth of each matrix
        rows_x_3d, cols_x_3d = random.randint(2, 5), random.randint(2, 5)
        rows_y_3d, cols_y_3d = cols_x_3d, random.randint(2, 5)
        
        x_3d = np.random.randint(1, 10, (batch_size, rows_x_3d, cols_x_3d))
        y_3d = np.random.randint(1, 10, (batch_size, rows_y_3d, cols_y_3d))

        # Apply the custom dot for each element in batch
        custom_result_3d = [dot(x_3d[i].tolist(), y_3d[i].tolist()) for i in range(batch_size)]
        numpy_result_3d = np.array([np.dot(x_3d[i], y_3d[i]) for i in range(batch_size)])
        
        # Compare for 3D matrices (batched dot product)
        assert np.allclose(custom_result_3d, numpy_result_3d.tolist()), f"Failed for 3D matrices with batch size {batch_size} x {rows_x_3d}x{cols_x_3d} and {rows_y_3d}x{cols_y_3d}"

    print("All tests passed!")

# Run the tests
test_dot_function()
