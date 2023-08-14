import random

# Generating and writing 100 data points
with open("sample_data.csv", "w") as file:
    file.write("x,f(x)\n")
    for _ in range(100):
        x = random.uniform(0, 1000)
        fx = 0.5*x**2 - 2*x + 1 + random.uniform(-100000, 100000)
        file.write(f"{x:.4f},{fx:.4f}\n")
