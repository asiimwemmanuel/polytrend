# For research purposes

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_multiplication(matrix_a, matrix_b):
    result = []
    for row in matrix_a:
        new_row = [dot_product(row, col) for col in transpose(matrix_b)]
        result.append(new_row)
    return result

def matrix_inversion(matrix):
    n = len(matrix)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for i in range(n):
        pivot = matrix[i][i]
        for j in range(n):
            matrix[i][j] /= pivot
            identity[i][j] /= pivot

        for k in range(n):
            if k != i:
                factor = matrix[k][i]
                for j in range(n):
                    matrix[k][j] -= factor * matrix[i][j]
                    identity[k][j] -= factor * identity[i][j]

    return identity

def normal_equation(X, y):
    X_with_bias = [[1] + row for row in X]
    X_transpose = transpose(X_with_bias)
    X_transpose_X = matrix_multiplication(X_transpose, X_with_bias)
    X_transpose_X_inv = matrix_inversion(X_transpose_X)
    X_transpose_X_inv_X_transpose = matrix_multiplication(X_transpose_X_inv, X_transpose)
    coefficients = matrix_multiplication(X_transpose_X_inv_X_transpose, [[target] for target in y])

    return [coeff[0] for coeff in coefficients]

# Example usage
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  # Feature matrix (4 data points with 2 features each)
y = [5, 8, 10, 13]  # Target vector

coefficients = normal_equation(X, y)
print("Coefficients:", coefficients)
