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

# Generating and writing 100 data points
with open("sample_data.csv", "w") as file:
    file.write("x,f(x)\n")
    for _ in range(100):
        x = random.uniform(0, 1000)
        fx = 0.5*x**2 - 2*x + 1 + random.uniform(-100000, 100000)
        file.write(f"{x:.4f},{fx:.4f}\n")
