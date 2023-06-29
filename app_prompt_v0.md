Linear regression

/* this will soon have all the specifics I need for PolyTrend final */
/* may have to use a ChatGPT plugin for explanations and analysis */

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
2. linear: *f(n) -> an + b*
2. polynomial: *f(n) -> a₀ + a₁n¹ + ... + axn^x*
3. exponential: *f(n) -> abⁿ*
4. logarithmic: *f(n) -> a + b ⨉ log(n)*
5. power: *f(n) -> a ⨉ n^b*
6. logistic: *f(n) -> L / 1 + e^-k(n - n₀)*
7. sinusoidal (via Fast Fourier Transform)

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