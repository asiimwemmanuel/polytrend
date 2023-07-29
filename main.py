import random
from typing import Union, List, Tuple
import numpy as np
# ! relative imports are buggy in test.py
from src.polytrend import PolyTrend

# * main serves as test due to relative imports bug, use the commented out import statement once it works
class PolyTrendTest():
	def __init__(self):
		self.polytrend = PolyTrend()

	def plotest(self, degree: Union[List[int], int]) -> None:
		# TODO Adjust the error bounds based on the degree so that the R-squared should be approximately 0.8.

		# abiding by PolyTrend.polyfind() format
		if isinstance(degree, int):
			degree = [degree]

		for deg in degree:
			data = self.generate_data(deg, 100, -100, 100, -10**(deg+1), 10**(deg+1))

			print("\nGenerating Data... 💭\n")
			for point in data:
				print(point)

			print("\nPlotting... 📈\n")
			self.polytrend.polyplot(degree, data)

		print("\nFinished! 😉\n")

	def generate_data(self, degree: int, num_samples: int, x_lower_bound: float, x_upper_bound: float, err_lower_bound: float, err_upper_bound: float) -> List[Tuple[float, float]]:
		# Generating a random polynomial of the specified degree; bounds are arbitrary
		coeffs = np.array([random.randint(1, 50) for i in range(degree)])

		# Generate the data using the specified parameters
		data = [
			(x, float(np.polyval(coeffs, x)) + random.uniform(err_lower_bound, err_upper_bound))
			for x in [
				random.uniform(x_lower_bound, x_upper_bound) for _ in range(num_samples)
			]
		]

		return data

if __name__ == '__main__':
	print("This is a sample of the PolyTrend capability 🕺")
	test = PolyTrendTest()
	test.plotest([3, 2, 1])
	print("Check ./images/ for the plots 🙈")
	print("\nThis is only a demo of PolyTrend... 👀")
	print("Feel free to fork this repo and use it in any projects & apps you're working on! 🤝")
