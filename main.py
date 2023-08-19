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

import sys
import pandas as pd
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from src.polytrend import PolyTrend
from gui.gui import PolyTrendUI

class PolyTrendApp(QMainWindow):
	def __init__(self):
		super().__init__()
		self.ui = PolyTrendUI()
		self.ui.setupUi(self)
		self.setWindowTitle("PolyTrend")  # Set the title of the window

		# Connect the 'Find' button click event to the plot_graph function
		self.ui.plot.clicked.connect(self.plot_graph)

		# Connect the 'Import CSV' button click event to the import_csv function
		self.ui.csv_button.clicked.connect(self.import_csv)

		# Enable/disable the 'Find' button based on the data in the text boxes
		self.ui.x_box.textChanged.connect(self.manage_find_button)
		self.ui.y_box.textChanged.connect(self.manage_find_button)
		self.ui.degree_box.textChanged.connect(self.manage_find_button)

	def enable_csv_path_box(self):
		# Enable the CSV path text box when the 'Import CSV' button is clicked
		self.ui.save_checkbox.setEnabled(True)

	def manage_find_button(self):
		x_values = self.ui.x_box.toPlainText().split()
		y_values = self.ui.y_box.toPlainText().split()
		degrees = self.ui.degree_box.toPlainText().split()

		# Enable the 'Find' button if there are at least two rows of data in x and y, and at least one row for degrees
		if len(x_values) >= 2 and len(y_values) >= 2 and len(degrees) >= 1:
			self.ui.plot.setEnabled(True)
		else:
			self.ui.plot.setEnabled(False)

	def import_csv(self):
		csv_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")

		if csv_path:
			# Load CSV file using the provided path
			df = pd.read_csv(csv_path)
			x_values = df[df.columns[0]].tolist()
			y_values = df[df.columns[1]].tolist()
			self.ui.x_box.setPlainText(" ".join(map(str, x_values)))
			self.ui.y_box.setPlainText(" ".join(map(str, y_values)))

			# Trigger the find button management
			self.manage_find_button()

	def plot_graph(self):
		# Get the values from the text boxes and convert to floats
		x_values = list(map(float, self.ui.x_box.toPlainText().split()))
		y_values = list(map(float, self.ui.y_box.toPlainText().split()))
		extrap = list(map(float, self.ui.extrap_box.toPlainText().split()))
		degrees = list(map(int, self.ui.degree_box.toPlainText().split()))

		# Create a PolyTrend object
		poly_trend = PolyTrend()

		# Plot the graph using the polyplot function while checking if the 'Save to PNG' checkbox is checked and save the figure if it is
		poly_trend.polyplot(degrees, list((zip(x_values, y_values))), extrap, self.ui.save_checkbox.isChecked())

if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = PolyTrendApp()
	window.show()
	sys.exit(app.exec())
