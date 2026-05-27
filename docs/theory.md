## Theory

### Quadratic Sequences

For a quadratic sequence conforming to the structure ( a, b, c ), wherein:

- ( \alpha = b - a )
- ( \beta = c - b )
- ( x = \beta - \alpha ),

it becomes evident that ( \alpha ) and ( \beta ) collectively engender a linear sequence, delineated as:

```math
x_n + (\alpha - x)
```

or equivalently:

```math
\alpha + \sum_{i=1}^{n-1} x
```

In delineating the foundational stratum (denoted as ( x )) within the context of the disparity tableau, it is apposite to attribute to it a degree of 0, thereby characterizing it as a constant. Consequently, the linear progression manifests as an amalgamation of ( x_i ) and the inaugural term. Indeed, the derivation of the nth term at any given degree necessitates a retrospective computation of the substratal strata. Employing analogous rationale, it becomes apparent that quadratic sequences (and, by extension, sequences of other orders) can be deduced by effectuating the summation of their primary term, denoted as ( a ), with a discernible combination of substratal variables.

Asiimwe's general form of quadratic nth term problems may be derived as:

```math
a + \sum_{i=1}^{n-1} \alpha + x(i-1)
```

The formulated expression is inherently tailored to address linear and quadratic sequences, thereby necessitating an alternative approach for sequences of higher degrees. Notably, upon scrutiny of Asiimwe's equation, a discernible pattern emerges. It is conjectured that said expression itself serves as the subject of summation for cubic dilemmas, while this newly derived expression assumes the role of the subject of summation for quartic predicaments. Evidently, a recursive pattern manifests within this methodology when extending its application to problems characterized by varying degrees.

A paragonal solution to the recursive conundrum at hand is conceivable, particularly when considering the nth sequence. Such a solution possesses a versatility that extends to sequences of complexities ranging from ( n-r ):

```math
\text{For } r \in \mathbb{Z},
```

1 \leq r \leq n-1

It is also intriguing to consider a framework wherein descending through the layers in this framework resembles differentiation, where each step involves deriving successive terms. Conversely, ascending through the layers mirrors integration, accumulating terms to reconstruct the original sequence. This analogy provides insight into the interplay of discrete elements, akin to calculus principles.

**HYPOTHESIS**: Upon representing the sequence with a polynomial, it is conjectured that the \( f^{(d)} \) derivative assumes a constant value, thereby equating the bottommost layer.

Efforts to exploit this property for constructing the `quadseq` via Calculus, with the aim of devising a comprehensive methodology applicable to problems of all degrees, encountered setbacks. Attempts to integrate (progress upwards) from the lower strata resulted in the loss of pertinent information, including constants and additional terms. Consequently, this led to alterations in the plots of the primary sequence and its integrated counterpart.

As of yet, the untested hypothesis to recover this lost information involves either scrutinizing the complete primary and integrated sequences (while acknowledging the transformation between them) to discern the directed (i.e., not absolute) phase in each sequence term, or identifying such a phase at each step of the integration process rather than solely at its culmination. Both approaches facilitate the retrieval of the missing information, aiding in the deduction of absent polynomial constants. However, it should be noted that this task poses greater challenges when addressing other types of lost function terms.

### Lagrange Interpolation

Lagrange interpolation, primarily employed for in-bound approximation, possesses properties conducive to solving nth-term problems without introducing error.

The Lagrange interpolation formula is expressed as:

```math id="mlkjb1"
P(x) = \sum_{i=0}^{n} \left( y_i \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j} \right)
```

Herein:

- ( P(x) ) denotes the polynomial of degree ( n ) (where ( n ) signifies the number of data points).
- ( y_i ) represents the y-coordinate of the ( i^{th} ) data point.
- ( x_i ) signifies the x-coordinate of the ( i^{th} ) data point.

This formula computes the lowest-order polynomial ( P(x) ) that traverses through the provided data points ( x_i, y_i ). It employs a weighted summation of Lagrange basis polynomials to interpolate the function or estimate the value at a specific point ( x ), where ( x ) corresponds to ( n ).

**Note**: The ( n ) in the formula does not correlate with the polynomial degree ( q ) referenced in the preceding section.

> For more information on the Lagrange polynomial, refer to the [Lagrange polynomial Wiki](https://en.wikipedia.org/wiki/Lagrange_polynomial).

### Polynomial Regression

Polynomial regression serves as a transition from discrete to continuous data, rendering it suitable for real-world data analytics scenarios characterized by inherent error. It offers a nuanced balance between accuracy and versatility across diverse applications.

In the context of a set comprising ( n ) data points ( (x, f(x)) ), the polynomial function of degree ( m ) is approximated as:

```math id="u6sdp4"
f(x_i) = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_m x_i^m + \varepsilon_i \quad \text{(for } i = 1, 2, \ldots, n \text{)}
```

In this formulation, ( m ) denotes the chosen maximum power, each ( \beta_i ) signifies a coefficient within the function, and ( \varepsilon_i ) represents random error.

The approximation is determined by solving for ( \vec{\beta} ) in the matrix equation:

```math id="9rkzo8"
\begin{bmatrix}
    1 & x_0 & x_0^2 & \dots & x_0^m \\
    1 & x_1 & x_1^2 & \dots & x_1^m \\
    1 & x_2 & x_2^2 & \dots & x_2^m \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & x_n & x_n^2 & \dots & x_n^m
\end{bmatrix}
\cdot
\begin{bmatrix}
    \beta_0\\
    \beta_1\\
    \beta_2\\
    \vdots \\
    \beta_m
\end{bmatrix}
+
\begin{bmatrix}
    \varepsilon_0\\
    \varepsilon_1\\
    \varepsilon_2 \\
    \vdots \\
    \varepsilon_n
\end{bmatrix}
=
\begin{bmatrix}
    y_0 \\
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
\end{bmatrix}
```

The matrix equation ( \mathbf{X} \vec{\beta} + \vec{\varepsilon} = \vec{y} ) is solved using the [Normal Equation](http://mlwiki.org/index.php/Normal_Equation):

```math id="z83pe3"
\widehat{\vec{\beta}} = (\mathbf{X}^\mathsf{T} \mathbf{X})^{-1} \mathbf{X}^\mathsf{T} \vec{y}
```

Where:

- ( \widehat{\vec{\beta}} ) is the estimated coefficient vector.
- ( \mathbf{X} ) is the matrix of input features.
- ( \vec{y} ) is the vector of target values.
- ( \mathbf{X}^\mathsf{T} ) denotes the transpose of the input feature matrix.
- ( (\mathbf{X}^\mathsf{T} \mathbf{X})^{-1} ) represents the inverse of the covariance matrix of the input features.

Note that the product of the transpose and inverse of the covariance of ( \mathbf{X} ) is also termed the [pseudoinverse](http://mlwiki.org/index.php?title=General_Inverse&action=edit&redlink=1) ( \mathbf{X^{+}} ), thus the coefficient vector is the product of the pseudoinverse and the target vector.

### Model Selection

The consistent tendency observed during the implementation of polynomial regression methods in predictive model generation is noteworthy. Specifically, when tasked with selecting the optimal degree for the model, typically between ( a ) and ( b ), the algorithm consistently opts for ( \max(a, b) ). This pattern persists as the algorithm discerns the superior degree to be that which yields a polynomial with a higher score, as gauged by a metric, often Mean Squared Error (MSE) in this context.

Intuitively, this phenomenon aligns with Occam's Razor principle, which posits that simpler solutions are often more preferable. Moreover, it is reasonable to expect programmers to streamline their algorithms to minimize unnecessary computations. Furthermore, this observation has likely been acknowledged within the field, as evidenced by the multitude of responses addressing similar occurrences, a commonality characteristic of such domains.

The Bayesian Information Criterion (BIC) is the most well known, and states:

```math id="glpdw5"
BIC = n \cdot \ln(MSE) + k \cdot \ln(n)
```

Where:

- ( n ) is the number of data points.
- ( MSE ) is the mean squared error.
- ( k ) is the number of parameters in the regression model.

```math id="tj1qw9"
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

Where:

- ( n ) is the number of data points.
- ( y_i ) is the actual value for the ( i )-th data point.
- ( \hat{y}\_i ) is the predicted value for the ( i )-th data point.

The allure of the Bayesian Information Criterion (BIC) lies in its dual function of rewarding accuracy, encapsulated within its first term in the summation, while simultaneously penalizing complexity through its second term. What distinguishes BIC is its additive nature, a departure from the multiplicative relationships prevalent in various scientifically derived quantities, such as velocity, among others. This peculiarity piqued my curiosity, prompting an inquiry into its derivation.

> For more info, see [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion)

## Additional Resources

A compilation of articles connected on the mentioned subjects

### Articles

[Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression).
[Regression Analysis](https://en.wikipedia.org/wiki/Regression_analysis)
[Interpolation](https://en.wikipedia.org/wiki/Interpolation)
[Bayesian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion)

### Videos

[Lagrange Interpolation](https://youtu.be/bzp_q7NDdd4?feature=shared)
[Polynomial Regression in Python](https://youtu.be/H8kocPOT5v0?feature=shared)
[Polynomial Regression in Python - sklearn](https://youtu.be/nqNdBlA-j4w?feature=shared)
