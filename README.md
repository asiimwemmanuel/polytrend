<!-- chatgpt, update the section labelled "Code Explanation: PolyTrend Class for Polynomial Trend Analysis" to include return types of the methods. -->

# Introduction

📈 **PolyTrend** is a regression algorithm that approximates and plots a polynomial function onto given data. It provides insights and conclusions in the fields of interpolation, and polynomial regression, specifically in the subfield of approximation theory. 💡

For more detailed information and comprehensive exploration, refer to the documentation available in the `./docs/` folder. 📚🔬

# Installation

To install PolyTrend, follow these steps:

1. Install [pip](https://pip.pypa.io/en/stable/installation/) if it is not already installed on your system. 💻⚙️

2. Clone the PolyTrend repository:

	```shell
	git clone https://github.com/asiimwemmanuel/polytrend.git
	```

3. Navigate to the root directory of the cloned repository:

	```shell
	cd polytrend
	```

4. Install the necessary dependencies:

	```shell
	pip install -r requirements.txt
	```
5. Run main (optional):
	```shell
	py main.py
	```

# Code Explanation: `PolyTrend` Class for Polynomial Trend Analysis

The code defines a class called `PolyTrend` that provides methods for polynomial trend analysis. Here's a summary of the code structure and its functionalities:

## Class Definition
The `PolyTrend` class encapsulates methods for polynomial trend analysis.

## Method: `polyplot`
This method plots the best polynomial fit for known data points. It takes the following parameters:
- `degrees`: A list of polynomial degrees to consider.
- `known_data`: A list of tuples representing the known data points.
- `extrap_data` (optional): A list of x coordinates for extrapolation.

## Method: `polyfind`
This method finds the best-fit polynomial function. It takes the following parameters:
- `degrees`: A list of polynomial degrees to consider.
- `known_data`: A list of tuples representing the known data points.
It returns a callable function that predicts values based on the polynomial.

## Method: `polygraph`
This method plots the function, known data points, and extrapolated data points. It takes the following parameters:
- `known_data`: A list of tuples representing the known data points.
- `extrap_data` (optional): A list of extrapolation data points.
- `func` (optional): A function to generate predicted values.
It raises a `ValueError` if no known data is provided or if extrapolation data is provided without a function.

## Error Handling
The code includes error handling to raise a `ValueError` if required parameters are missing or invalid.

## File Saving
The code saves the generated plot as a PNG file in a directory based on the current timestamp.

# Theory

## Quadratic Sequences

A quadratic sequence follows the form: $a, b, c$

If $α = b - c, β = c - b$ and $x = β - α$

It can be observed that $α$ and $β$ form a linear sequence, represented as:

$$xn + (\alpha - x)$$

This can also be written as:

$$\alpha + \sum_{i=1}^{n-1}x$$

Since the third layer (x) can be seen as a 'constant' sequence (in relation to degree), the linear sequence can be seen as a summation of the constant sequence with the first term. After all, finding the nth term at any degree involves computation of the underlying layers.

With this same logic, quadratic sequences (and any other sequences) can be computed as their first term a summed with the summation of the general form of the closest underlying sequence:

$$a + \sum_{i=1}^{n-1}(x_i + \alpha-x)$$

For quadratic sequences, the number of summations/computations increases linearly with the degree of the sequence (e.g., 1 for quadratic, 2 for cubic, and so on). However, the above formula is limited to linear and quadratic sequences, requiring a different method for higher degrees.

It's also interesting to think about traversing down the layers as differentiation and traversing up the layers as integration. 🔍🔢

## Lagrange Interpolation

Lagrange interpolation is a method primarily used for in-bound approximation. However, its properties make it suitable for nth-term problems, particularly those without error.

The Lagrange interpolation formula is given by:

$$P(x)=\sum_{i=0}^{n}\left(y_i\prod_{j=0,j\neq i}^{n}\frac{x-x_j}{x_i-x_j}\right)$$

Where:
- $P(x)$ represents the polynomial of degree n (where n is the number of data points minus 1).
- $y_i$ denotes the y-coordinate of the ith data point.
- $x_i$ represents the x-coordinate of the ith data point.

This formula calculates the polynomial $P(x)$ that passes through the given data points $x_i, y_i$. It uses a weighted sum of Lagrange basis polynomials to interpolate the function or estimate the value at a specific point x, where x represents n.

For more information on Lagrange polynomial, refer to the [Lagrange polynomial Wiki](https://en.wikipedia.org/wiki/Lagrange_polynomial). In PolyTrend, error resistance is incorporated with a new technique. 📈🔍

## Polynomial Regression

Polynomial regression is used to shift from discrete to continuous data, making it applicable in real-world data analytics where error is present. It offers a tradeoff between accuracy and generality for various applications.

Given a set of data points $(x, f(x))$, the polynomial function is approximated as:

$$f(x) \approx \beta_0 + \beta_1x + \beta_2x^2 + ...+ \beta_qx^q + \varepsilon$$

Each $\beta$ represents a coefficient in the function, and $\varepsilon$ represents random error.

The approximation is determined by solving the equation:

$$\begin{bmatrix}\beta_0 \\ \beta_1 \\ ... \\ \beta_n \\ \end{bmatrix} \cdot \begin{bmatrix}1 & x_0 & x_0^2 & ... & x_0^q \\ 1 & x_1 & x_1^2 & ... & x_1^q \\ ... & ... & ... & ... & ...\\ 1 & x_n & x_n^2 & ... & x_n^q\\ \end{bmatrix} \approx \begin{bmatrix}y_0 \\ y_1 \\ ... \\ y_n \\ \end{bmatrix}$$

The matrix equation $BX \approx Y$ is solved using the equation $B \approx YX^T(XX^T)^{-1}$, where $B$ represents the matrix of coefficients.

For further information on polynomial regression, refer to the [polynomial regression wiki](https://en.wikipedia.org/wiki/Polynomial_regression). 📈🔢

## Future Improvements
PolyTrend will utilize K-fold cross-validation to evaluate the models.

The PolyTrend class can benefit from the following future improvements:

- **Documentation**: Enhance existing docstrings with detailed explanations and examples to improve clarity and usability.

- **Input Validation**: Expand input validation to handle edge cases and provide clear error messages.

- **Exception Handling**: Catch specific exceptions when saving the plot for better error handling.

- **Flexibility**: Add an option to plot individual polynomial fits for each degree.

- **Performance Optimization**: Optimize the `polyfind` method for improved computational efficiency.

- **Modularity**: Split the code into smaller functions or methods for better modularity and maintainability.

As the developer of PolyTrend, consider exploring advanced optimization and selection methods beyond the brute force approach with R-squared. These methods can improve efficiency and accuracy, especially for scenarios with limited computational resources or a need for faster model selection.

Techniques such as stepwise regression, Lasso, or Ridge regression can be leveraged to automatically select important features or control model complexity, providing a streamlined model selection process. Additionally, information criteria such as AIC or BIC offer a balanced approach by penalizing excessive parameters and preventing overfitting.

Choosing the appropriate model selection method depends on specific requirements and constraints. While the current brute force approach with R-squared suffices for many use cases, adopting more advanced methods can enhance efficiency and flexibility.

Future enhancements should consider factors such as computational resources, the need for faster model selection, and the trade-off between model complexity and generalization performance. Exploring advanced optimization and selection methods will further refine the PolyTrend model selection process. ⚙️📊

These improvements will enhance the code's usability, performance, and maintainability, providing a better experience for engineering and data science students using the PolyTrend class.

Feel free to add to this repo with your own specialized tools according to your preference and expertise 😉