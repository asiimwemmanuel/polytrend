import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from src.polytrend import PolyTrend
from gui.gui import PolyTrendUi

class PolyTrendApp(QMainWindow):
	def __init__(self):
		super().__init__()
		self.ui = PolyTrendUi()
		self.ui.setupUi(self)
		# TODO - set the title of the window to 'PolyTrend' - it's just MainWindow right now

		# Connect the 'Find' button click event to the plot_graph function
		self.ui.plot.clicked.connect(self.plot_graph)

		# Enable/disable the 'Find' button based on the data in the text boxes
		self.ui.x_box.textChanged.connect(self.manage_find_button)
		self.ui.y_box.textChanged.connect(self.manage_find_button)
		self.ui.degree_box.textChanged.connect(self.manage_find_button)

	def manage_find_button(self):
		x_values = self.ui.x_box.toPlainText().split()
		y_values = self.ui.y_box.toPlainText().split()
		degrees = self.ui.degree_box.toPlainText().split()

		# Enable the 'Find' button if there are at least two rows of data in x and y, and at least one row for degrees
		if len(x_values) >= 2 and len(y_values) >= 2 and len(degrees) >= 1:
			self.ui.plot.setEnabled(True)
		else:
			self.ui.plot.setEnabled(False)

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
