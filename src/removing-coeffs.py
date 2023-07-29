import numpy as np

def measure_coefficient_contributions(coefficients):
    total_sum = np.sum(np.abs(coefficients))
    contributions = np.abs(coefficients) / total_sum
    return contributions

def eliminate_small_coefficients(coefficients, threshold=0.01):
    contributions = measure_coefficient_contributions(coefficients)
    filtered_coefficients = np.where(contributions < threshold, 0.0, coefficients)
    return filtered_coefficients

# Example usage
coefficients = np.array([0.1, 0.05, 0.02, 0.001, 0.3, 0.15])
filtered_coefficients = eliminate_small_coefficients(coefficients, threshold=0.01)

print("Original Coefficients:", coefficients)
print("Filtered Coefficients:", filtered_coefficients)
