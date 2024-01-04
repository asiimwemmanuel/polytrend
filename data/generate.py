# This file is part of PolyTrend.
#
# PolyTrend is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PolyTrend is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PolyTrend. If not, see <https://www.gnu.org/licenses/>.

import random

# Generating and writing 1000 data points
with open("sample_data.csv", "w") as file:
    file.write("x,f(x)\n")
    for _ in range(0, 100):
        x_var = random.uniform(-1000, 1000)
        y_var = -0.1*x_var**3 + x_var**2 + 0.2*x_var + 10 + random.uniform(-10000*10000, 10000*10000)
        file.write(f"{x_var:.4f},{y_var:.4f}\n")
