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