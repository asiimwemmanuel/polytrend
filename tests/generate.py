import csv
import random

# Underlying function f(x) = 2x^2 - 5x + 3
def f(x):
	return 2 * x**2 - 5 * x + 3

# Generate sample data with random error
def generate_sample_data(n=100):
	data = []
	for i in range(n):
		x = random.uniform(-100, 100)
		y = f(x) + random.uniform(-5000, 5000)  # Adding random noise between -5000 and 5000
		data.append((x, y))
	return data

# Save sample data to a CSV file
def save_to_csv(filename, data):
	with open(filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for x, y in data:
			writer.writerow([x, y])

if __name__ == "__main__":
	sample_data = generate_sample_data(1000)
	save_to_csv('sample_data.csv', sample_data)
