import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def poly(data, custom_degree=None):
    x = np.array([point[0] for point in data]).reshape(-1, 1)
    y = np.array([point[1] for point in data])

    if custom_degree is not None:
        degree_range = [custom_degree]
    else:
        degree_range = range(5)

    best_degree = None
    best_mse = float('inf')
    best_model = None

    for degree in degree_range:
        polynomial_features = PolynomialFeatures(degree=degree)
        x_poly = polynomial_features.fit_transform(x)

        model = LinearRegression()
        model.fit(x_poly, y)

        y_pred = model.predict(x_poly)
        mse = mean_squared_error(y, y_pred)

        if mse < best_mse:
            best_degree = degree
            best_mse = mse
            best_model = model

    coefficients = best_model.coef_
    intercept = best_model.intercept_

    def polynomial_expression(x):
        expression = f"{intercept:.2f}"
        for i, coeff in enumerate(coefficients[0][1:], start=1):
            expression += f" + {coeff:.2f} * x^{i}"
        return expression

    print(f"Best degree: {best_degree}")
    print(f"Mean squared error: {best_mse}")
    print(f"Polynomial expression: f(x) = {polynomial_expression('x')}")
    
    # return {
    #     'best_degree': best_degree,
    #     'mean_squared_error': best_mse,
    #     'polynomial_expression': polynomial_expression('x'),
    #     'best_model': best_model
    # }

    return lambda x: best_model.predict(polynomial_features.transform(x))


def graph(function, known_data, extrapolated_data):
    x_known = np.array([point[0] for point in known_data])
    y_known = np.array([point[1] for point in known_data])
    x_extrapolated = np.array(extrapolated_data)

    plt.scatter(x_known, y_known, color='blue', label='Known Data')
    plt.scatter(x_extrapolated, function(x_extrapolated), color='red', label='Extrapolated Data')
    plt.plot(x_known, y_known, color='blue', label='Polynomial Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def polyplot(known_data, extrapolated_data, custom_degree=None):
    function = poly(known_data, custom_degree)
    graph(function, known_data, extrapolated_data)
