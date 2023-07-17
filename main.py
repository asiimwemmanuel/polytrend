# ! main serves as test due to relative imports bug, use the commented out import statement once it works

import random
from typing import List, Tuple
import numpy as np

# ! relative import is buggy in test file. decided to use it here
from src.polytrend import PolyTrend

class PolyTrendTest():
    def __init__(self):
        self.polytrend = PolyTrend()

    def plotest(self, degree: int) -> None:
        # Adjust the error bounds based on the degree. The desired R-squared should be approximately 0.8.
        data = self.generate_data(degree, 100, -100, 100, -100**(degree), 100**(degree))
        print("\nGenerated Data:\n")
        for point in data:
            print(point)
        print("\nPlotting...\n")
        self.polytrend.polyplot([degree], data)  # Pass the degree as a list, not a single integer
        print("\nFinished 😉\n")

    def generate_data(self, degree: int, num_samples: int, data_lower_bound: float, data_upper_bound: float, err_lower_bound: float, err_upper_bound: float) -> List[Tuple[float, float]]:
        # Generating a random polynomial of the specified degree; bounds are arbitrary
        coeffs = np.array([random.randint(1, 10) for i in range(degree+1)])

        # Generate the data using the specified parameters
        data = [
            (x, float(np.polyval(coeffs, x)) + random.uniform(err_lower_bound, err_upper_bound))
            for x in [
                random.uniform(data_lower_bound, data_upper_bound) for _ in range(num_samples)
            ]
        ]

        return data

if __name__ == '__main__':
    print("This is a simple example of the PolyTrend capability 🕺")
    test = PolyTrendTest()
    for i in range(1, 5):
        test.plotest(i)
    print("Check ./models for the plots 😁")
    print("\nThis is only a demo of PolyTrend. Feel free to fork this repo and use it in any projects/apps you're working on! 🤝")
