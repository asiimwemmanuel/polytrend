I have an app called PolyTrend to be used on desktop and mobile on Windows, macOS & Linux as well as Android & iOS respectively.
It's meant to find relationships and fit curves onto data.

Below are the specifics of the desktop version.

# General UI

'PolyTrend' heading
Small "intro"; an explanation of the app and its functions
Drop down menu for function type (options are listed below)
There's two input text boxes: known data and points to extrapolate
Graph/model, plotting the relationship in the known data
Short rundown of the relationship: how it was calculated and why that fit was chosen

(also include a light/dark theme option)

# Possible function types

1. auto
2. linear: _f(n) -> an + b_
3. polynomial: _f(n) -> a₀ + a₁n¹ + ... + axn^x_
4. exponential: _f(n) -> abⁿ_
5. logarithmic: _f(n) -> a + b ⨉ log(n)_
6. power: _f(n) -> a ⨉ n^b_
7. logistic: _f(n) -> L / 1 + e^-k(n - n₀)_
8. sinusoidal (via Fast Fourier Transform)

'auto' is where the app will automatically pick the best curve fit among the listed types ie. the fit with the least error

# Other

there are two text boxes for x, y coordinates where each row represents a single point.
make sure in the margin, indicate the point eg. point 1, point 2 ... for each row
the user can input any amount of data they wish
for extrapolates, the user inputs a list of x values that the fitted curve will estimate their values.

# Use case

1. data points and extrapolates are provided
2. type or relationship is given
3. graph is plotted (with short explanation at the bottom), distinguishing between given and calculated data points
4. include model description (with equation), assumptions, goodness of fit (with R-squared and RMSE and other statistical measures), coefficient estimates and discretions, diagnostic plots, validation routines used and any recommendations

# Backend

make a separate function for each of the possible function types, with the auto function utilizing all the others.
make each function type output a lambda function such that the following example holds:

data = [(x,y), (x,y) ...] # 2d list of tuples
fitted_func = auto(data)
print(fitted_func(x_value)) # prints the fitted function's evaluation at the x_value

all the above apply for mobile, only scaled to the screens size and navigation of the device.

# Disclaimer: This is a ChatGPT instruction meant to guide the AI on alg_v3's functions

Make a Python script that implements a polynomial regression model on a given list of tuples as data for degrees 0 through 4 inclusive and selects the most accurate model via MSE.

It should plot a graph headed "Fitted Function via PolyTrend" and plots the model with a viewport of 150% of the size of the input data, differentiating between known vs calculated data with labels (which is given as a 2nd parameter list).

The script also outputs the algebraic expression of the polynomial in the form
_f(x) = a_n _ x^n + a\_(n-1) _ x^(n-1) + ... + a_2 _ x^2 + a*1 * x + a*0*

This script must have 2 functions in the parent class polytrend, namely poly (1 parameter) and graph (3 parameters). poly returns a lambda function such that x = poly(data), x(4)

Is valid and returns the value of the function at x = 4.

graph simply takes in data (list of tuples), extrapolates (list of x values) and a function. it plots the function and indicates the known vs extrapolated data.
