<!-- chatgpt should update this and all other documentation to suit the software -->
<!-- next part of development: register on PyPI, make docker image, generate binaries -->

# PolyTrend 📈

## Introduction 💡

PolyTrend is a regression algorithm that approximates and plots a polynomial function onto given data. It provides insights and conclusions in the fields of interpolation and polynomial regression, specifically in the subfield of approximation theory.

> **Note**:
> For more detailed information and comprehensive exploration, refer to the documentation available in the `./docs/` folder. 📚🔬

## Installation ⚙️

<!-- methods to include: source (suitable for modification/customization), python package index (suitable for usage in projects), docker image (suitable for running in a VM), installation wizard with binary(suitable for standalone releases) -->

### From source

To install PolyTrend **from source**, follow these steps:

- Install [Python](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads) if not already installed on your system. 💻

- Set up the environment (clone repo, install reqs, sample the runs)

```shell
git clone https://github.com/asiimwemmanuel/polytrend.git
cd polytrend
pip install -r requirements.txt
```

- Run the app

```shell
py main.py
```

> Note for linux: If issues persist, visit [this website](https://web.stanford.edu/dept/cs_edu/resources/qt/install-linux) and follow the instructions to configure Qt on your system

<!-- ### From the python packaging index * -->

<!-- To install polytrend **as a package**, follow these instructions:

- Install [Python](https://www.python.org/downloads/) if it is not already installed on your system. 💻

- Install the package:

```shell
pip install polytrend
```
-->

<!-- ### From a docker image

This will soon be available... -->

<!-- ### From a standalone release/binary

To install polytrend via a wizard, see [Releases](https://github.com/asiimwemmanuel/polytrend/releases)-->

<!-- Note: If you are using WSL or a GUI incapable OS, install an X11 server like vcxsrv and link your forwarding instance with your terminal session.
Do so only if the app fails to run in case you installed it from source -->

## Code Explanation: `PolyTrend` Class for Polynomial Trend Analysis 🧮

The code defines a class called `PolyTrend` that provides methods for polynomial trend analysis. Here's a summary of the code structure and its functionalities:

### Class Definition

The `PolyTrend` class encapsulates methods for polynomial trend analysis.

### Method: `polyplot` 📈

This method plots the best polynomial fit for known data points. It takes the following parameters:

- `degrees`: A list of polynomial degrees to consider.
- `known_data`: A list of tuples representing the known data points.
- `extrap_data` (optional): A list of x coordinates for extrapolation.

### Method: `polyfind` 🔍

This method finds the best-fit polynomial function. It takes the following parameters:

- `degrees`: A list of polynomial degrees to consider.
- `known_data`: A list of tuples representing the known data points.
- It returns a callable function that predicts values based on the polynomial.

### Method: `polygraph` 📊

This method plots the function, known data points, and extrapolated data points. It takes the following parameters:

- `known_data`: A list of tuples representing the known data points.
- `extrap_data` (optional): A list of extrapolation data points.
- `func` (optional): A function to generate predicted values.

### Error Handling ❗️

The code includes error handling to raise a `ValueError` if required parameters are missing or invalid.

### File Saving 💾

The code saves the generated plot as a PNG file in the `./models` directory with a timestamp and a random integer between 0 and 1000.

## Theory 🔍🔢

### Quadratic Sequences

If a quadratic sequence follows the form:

$a, b, c$

And while letting;

- $α = b - c$
- $β = c - b$
- $x = β - α$

It can be observed that $α$ and $β$ form a linear sequence, represented as:

$$xn + (\alpha - x)$$

or

$$\alpha + \sum_{i=1}^{n-1}x$$

Taking the bottommost layer (x) of the difference table to have a degree of 0 i.e constant, the linear sequence can be seen as a summation of $x_i$ and the first term. After all, finding the nth term at any degree involves regressive computation of the underlying layers.
With this same logic, quadratic sequences (and any other, for that matter) can be computed as their first term $a$ summed with a certain combination of underlying variables.

Asiimwe's general form of quadratic nth term problems:

$$a + \sum_{i=1}^{n-1}\alpha + x(i-1)$$

<!-- also comment about how this ties into Sequences and series, and general formulae for calculating summations -->

The above formula is limited to linear and quadratic sequences, requiring a different method for higher degrees. Notice the general form of linear sequences as the subject of summation. It can be inferred that that expression itself is the subject of summation for cubic problems and THAT expression is the subject of quartic problems. A recursive pattern is observed in this approach when generalizing for problems with variable degrees.

It's also interesting to think about traversing the layers downwards as differentiation and upwards as integration.

**HYPOTHESIS**: when the sequence is represented with a polynomial, the $f^{(d)}$ derivative is a constant equating the bottommost layer.

My attempts to use this property to build `quadseq` via Calculus thereby forming a general method for all degree problems fell short as attempting to integrate (traverse upwards) from the lower layers led to the loss of information (constants and extra terms), causing a transformation in the plots of the main sequence and the integrated one.

My untested hypothesis to regain this information so far is either to look at the full main and integrated sequences (while noticing one is a transformation of the other) and find the directed (that is, not absolute) phase in each sequence term or find such a phase at each integral of the process, rather at the end; both help in finding the missing information. This is best to deduce missing polynomial constants and is more challenging for other types of lost function terms.

<!-- TODO: include an example of the hypothesis -->

### Lagrange Interpolation

Lagrange interpolation is a method primarily used for in-bound approximation. However, its properties make it suitable for nth-term problems as they are without error.

The Lagrange interpolation formula is given by:

$$P(x)=\sum_{i=0}^{n}\left(y_i\prod_{j=0,j\neq i}^{n}\frac{x-x_j}{x_i-x_j}\right)$$

Where:

- $P(x)$ represents the polynomial of degree $n$ (where $n$ is the number of data points). <!-- ! check for error... -->
- $y_i$ denotes the y-coordinate of the $i^{th}$ data point.
- $x_i$ represents the x-coordinate of the $i^{th}$ data point.

This formula calculates the lowest order polynomial $P(x)$ that passes through the given data points $x_i, y_i$. It uses a weighted sum of Lagrange basis polynomials to interpolate the function or estimate the

 value at a specific point $x$, where $x$ represents $n$.

**Note**: $n$ in the formula is not the same as the polynomial degree $q$ used in the previous section.

For more information on the Lagrange polynomial, refer to the [Lagrange polynomial Wiki](https://en.wikipedia.org/wiki/Lagrange_polynomial).

To increase error tolerance and applicability in real-world data, a new technique is incorporated. 📈🔍

### Polynomial Regression

Polynomial regression is used to shift from discrete to continuous data, making it applicable in real-world data analytics where error is present. It offers a tradeoff between accuracy and generality for various applications.

Given a set of $n$ data points $(x, f(x))$, the polynomial function is approximated as:

$$f(x) \approx \beta_0 + \beta_1x + \beta_2x^2 + ...+ \beta_px^p + \varepsilon$$

Where $p$ is the decided maximum power. Each $\beta_i$ represents a coefficient in the function, and $\varepsilon$ represents random error.

The approximation is determined by solving the equation:

$$\begin{bmatrix}1 & x_0 & x_0^2 & ... & x_0^p \\ 1 & x_1 & x_1^2 & ... & x_1^p \\ \vdots & \vdots & \vdots & \ddots & \vdots\\ 1 & x_n & x_n^2 & ... & x_n^p\\ \end{bmatrix} \cdot \begin{bmatrix}\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \\ \end{bmatrix} \approx \begin{bmatrix}y_0 \\ y_1 \\ \vdots \\ y_n \\ \end{bmatrix}$$

The matrix equation $XB \approx Y$ is solved using the [*Normal Equation*](https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/): $$B \approx (X^{T}X)^{-1}(X^{T}Y)$$

Where:

- $B$ is the coefficient vector.
- $X$ is the matrix of input features.
- $Y$ is the vector of target values.
- $X^{T}$ denotes the transpose of the input feature matrix.
- $(X^{T}X)^{-1}$ represents the inverse of the covariance matrix of the input features.

Dimensions of the matrices:

- $X$ is $(n+1)$ by $(p+1)$,
- $B$ is $(p+1)$ by $1$,
- $Y$ is $(n+1)$ by $1$.

> **Note**:
> For further information on polynomial regression, refer to the [Polynomial regression Wiki](https://en.wikipedia.org/wiki/Polynomial_regression). 📈🔢

## Future Improvements 🔮🔧

Here are some possible improvements for the script:

🚀 **Possible Improvements for the Script:**

1. **Batch Plotting:** 🔄 Create a separate function to perform polynomial regression and plot the results, to avoid redundant computations.

2. **Caching:** 🗄️ Implement caching to store regression results for different degrees and known data, speeding up repeated calculations.

3. **Data Normalization:** 📏 Normalize the data before regression to ensure consistent scaling and improve model convergence.

4. **Regularization:** 🎛️ Add regularization terms (L1/Lasso or L2/Ridge) to the regression model to prevent overfitting.

5. **Model Evaluation:** 📊 Consider additional metrics (e.g., MAE, MSE) for a comprehensive assessment of model performance.

6. **Error Handling:** ❗ Enhance error messages and gracefully handle edge cases to improve script robustness.

7. **Optimize Graph Prerequisites:** 📈 Fine-tune graph settings for better visualization and aesthetics.

8. **Parallelization:** 🏋️‍♀️ Explore parallelization to distribute computational load for large datasets.

9. **Outlier Handling:** 🚨 Address potential outliers in the data for more robust regression results.

10. **Feature Selection:** 🎯 Investigate feature selection techniques to identify the most relevant predictors.

11. **Cross-Validation:** 🔄 Implement cross-validation to assess model generalization and performance.

12. **Multivariate Regression:** 🧮 Extend the script for multivariate regression scenarios.

13. **Data Preprocessing:** 🛠️ Improve data preprocessing techniques for cleaner data inputs.

14. **Documentation:** 📝 Enhance code comments and provide detailed documentation for future maintainability.

15. **Performance Profiling:** 🛠️ Use tools like `cProfile` to identify performance bottlenecks and optimize critical sections.

16. **Model Selection:** 🤔 Compare different regression models (e.g., polynomial, linear, etc.) for the best fit.

17. **Feature Engineering:** 🔧 Explore feature engineering to create more informative predictors.

18. **Visualizations:** 📊 Create additional informative visualizations for data exploration and model evaluation.

19. **Ensemble Models:** 🤝 Consider ensemble methods (e.g., stacking, bagging) to improve prediction accuracy.

20. **Hyperparameter Tuning:** 🔍 Fine-tune model hyperparameters to optimize performance.

The relevance of these improvements depends on the specific use case and data characteristics. Prioritize the changes based on the requirements and constraints of your problem to achieve the best possible results. 🌟

Feel free to add to this repo with your specialized tools according to your preference and expertise 😉 🚀
