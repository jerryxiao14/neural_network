import math


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


def sigmoid(x):
    # Activation function for sigmoid is f(x) = 1/(1+e^(-x))
    if isinstance(x,(int,float)):
        return 1/(1+math.exp(-x))
    
    if isinstance(x,list):
        return [sigmoid(elem) for elem in x]

def relu(x):
    """
    ReLU activation function.
    f(x) = max(0, x)
    """
    if isinstance(x, (int, float)):
        return max(0, x)
    
    if isinstance(x, list):
        return [relu(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [relu(row) for row in x]

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    f(x) = max(alpha * x, x)
    """
    if isinstance(x, (int, float)):
        return max(alpha * x, x)
    
    if isinstance(x, list):
        return [leaky_relu(elem, alpha) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [leaky_relu(row, alpha) for row in x]

def elu(x, alpha=1.0):
    """
    ELU activation function.
    f(x) = x for x > 0, alpha * (exp(x) - 1) for x <= 0
    """
    if isinstance(x, (int, float)):
        return x if x > 0 else alpha * (math.exp(x) - 1)
    
    if isinstance(x, list):
        return [elu(elem, alpha) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [elu(row, alpha) for row in x]

def tanh(x):
    """
    Tanh activation function.
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    if isinstance(x, (int, float)):
        return math.tanh(x)
    
    if isinstance(x, list):
        return [tanh(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [tanh(row) for row in x]

def softmax(x):
    """
    Softmax activation function (only works on lists or 2D matrices).
    f(x)_i = exp(x_i) / sum(exp(x_j) for all j)
    """
    if isinstance(x, list):
        exp_x = [math.exp(i) for i in x]
        total = sum(exp_x)
        return [i / total for i in exp_x]

    if isinstance(x, list) and isinstance(x[0], list):
        return [softmax(row) for row in x]

def swish(x):
    """
    Swish activation function.
    f(x) = x * sigmoid(x)
    """
    if isinstance(x, (int, float)):
        return x * (1 / (1 + math.exp(-x)))
    
    if isinstance(x, list):
        return [swish(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [swish(row) for row in x]

def gelu(x):
    """
    GELU activation function.
    f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    
    if isinstance(x, list):
        return [gelu(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [gelu(row) for row in x]

def softplus(x):
    """
    Softplus activation function.
    f(x) = ln(1 + exp(x))
    """
    if isinstance(x, (int, float)):
        return math.log(1 + math.exp(x))
    
    if isinstance(x, list):
        return [softplus(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [softplus(row) for row in x]

def hard_sigmoid(x):
    """
    Hard Sigmoid activation function (simplified version of sigmoid).
    f(x) = clip((x + 1) / 2, 0, 1)
    """
    if isinstance(x, (int, float)):
        return min(max((x + 1) / 2, 0), 1)
    
    if isinstance(x, list):
        return [hard_sigmoid(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [hard_sigmoid(row) for row in x]

def hard_swish(x):
    """
    Hard Swish activation function.
    f(x) = x * clip(x + 3, 0, 6) / 6
    """
    if isinstance(x, (int, float)):
        return x * min(max(x + 3, 0), 6) / 6
    
    if isinstance(x, list):
        return [hard_swish(elem) for elem in x]
    
    if isinstance(x, list) and isinstance(x[0], list):
        return [hard_swish(row) for row in x]
