import math

def mean_squared_error(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # to avoid log(0)
    y_pred = [max(min(yp, 1 - epsilon), epsilon) for yp in y_pred]
    return -sum(yt * math.log(yp) + (1 - yt) * math.log(1 - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Example usage:
y_true = [1, 0, 1, 1]
y_pred = [0.9, 0.1, 0.8, 0.7]

print("MSE:", mean_squared_error(y_true, y_pred))
print("Cross-Entropy Loss:", cross_entropy_loss(y_true, y_pred))


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # to avoid log(0)
    y_pred = [max(min(yp, 1 - epsilon), epsilon) for yp in y_pred]
    return -sum(yt * math.log(yp) + (1 - yt) * math.log(1 - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

def cross_entropy_loss_derivative(y_true, y_pred):
    epsilon = 1e-15  # to avoid division by zero
    y_pred = [max(min(yp, 1 - epsilon), epsilon) for yp in y_pred]
    return [-(yt / yp) + (1 - yt) / (1 - yp) for yt, yp in zip(y_true, y_pred)]