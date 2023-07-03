# DISCLAIMER: THIS IS A CHATGPT INSTRUCTION MEANT TO GUIDE THE A.I. ON SOME OF THE ALGORITHM'S FUNCTIONS.

Make a Python script that implements a polynomial regression model on a given list of tuples as data for degrees 0 through 4 inclusive and selects the most accurate model via MSE.

It should plot a graph headed "Fitted Function via PolyTrend" and plots the model with a viewport of 150% of the size of the input data, differentiating between known vs calculated data with labels (which is given as a 2nd parameter list).

The script also outputs the algebraic expression of the polynomial in the form
*f(x) = a_n * x^n + a_(n-1) * x^(n-1) + ... + a_2 * x^2 + a_1 * x + a_0*

This script must have 2 functions in the parent class polytrend, namely poly (1 parameter) and graph (3 parameters). poly returns a lambda function such that x = poly(data), x(4)

Is valid and returns the value of the function at x = 4.

graph simply takes in data (list of tuples), extrapolates (list of x values) and a function. it plots the function and indicates the known vs extrapolated data.