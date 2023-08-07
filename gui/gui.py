import sys
from PySide6.QtWidgets import QApplication, QCheckBox, QGridLayout, QLabel, QMainWindow, QPushButton, QStatusBar, QTextEdit, QVBoxLayout, QWidget
class PolyTrendUI(object):
    def setupUi(self, PolyTrend):
        PolyTrend.setObjectName("PolyTrend")
        PolyTrend.resize(398, 356)
        self.centralwidget = QWidget(PolyTrend)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.heading = QLabel(self.centralwidget)
        self.intro = QLabel(self.centralwidget)
        self.label_6 = QLabel(self.centralwidget)
        self.label_5 = QLabel(self.centralwidget)
        self.label_4 = QLabel(self.centralwidget)
        self.label_3 = QLabel(self.centralwidget)
        self.heading.setWordWrap(True)

        self.heading.setText("This is PolyTrend - a simple clean way to create polynomial fits in data")
        self.intro.setText("For csv data, your data MUST be in the form x,f(x).\n"
                          "Make sure the first row (of the CSV) are axis titles.\n"
                          "It is assumed that values on the same row correspond to the same point.\n"
                          "Note that CSV data does not apply to extrapolate points.")
        self.label_6.setText("extrapolates\n(optional)")
        self.label_5.setText("degrees")
        self.label_4.setText("f(x) values")
        self.label_3.setText("x values")
        self.degree_box = QTextEdit(self.centralwidget)
        self.extrap_box = QTextEdit(self.centralwidget)
        self.x_box = QTextEdit(self.centralwidget)
        self.y_box = QTextEdit(self.centralwidget)
        self.csv_path_box = QTextEdit(self.centralwidget)  # Add a new QTextEdit for the CSV path

        self.degree_box.setMinimumSize(41, 121)
        self.x_box.setMinimumSize(61, 121)
        self.y_box.setMinimumSize(61, 121)
        self.csv_path_box.setMinimumSize(300, 30)  # Set the size of the CSV path box

        self.degree_box.setPlaceholderText("Enter degrees (e.g., 1 2 3)")
        self.extrap_box.setPlaceholderText("Enter extrapolation values")
        self.x_box.setPlaceholderText("Enter x values")
        self.y_box.setPlaceholderText("Enter f(x) values")
        self.csv_path_box.setPlaceholderText("Paste CSV file path here")  # Set a placeholder for the CSV path box

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.gridLayout.addWidget(self.degree_box, 0, 5, 1, 1)
        self.gridLayout.addWidget(self.extrap_box, 0, 7, 1, 1)
        self.gridLayout.addWidget(self.label_6, 0, 6, 1, 1)
        self.gridLayout.addWidget(self.label_5, 0, 4, 1, 1)
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.x_box, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.y_box, 0, 3, 1, 1)
        self.gridLayout.addWidget(self.csv_path_box, 1, 0, 1, 8)  # Add the CSV path box to the layout

        self.verticalLayout.addLayout(self.gridLayout)

        self.csv_button = QPushButton(self.centralwidget)
        self.plot = QPushButton(self.centralwidget)
        self.save_checkbox = QCheckBox(self.centralwidget)
        self.csv_button.setToolTip("Import CSV file")
        self.plot.setToolTip("Fit polynomial and plot")

        # Set tooltip text and button labels directly
        self.csv_button.setToolTip("Import CSV file")
        self.plot.setToolTip("Fit polynomial and plot")
        self.csv_button.setText("Import csv file")
        self.plot.setText("Fit polynomial")
        self.save_checkbox.setText("Save to PNG")

        self.verticalLayout.addWidget(self.csv_button)
        self.verticalLayout.addWidget(self.plot)
        self.verticalLayout.addWidget(self.save_checkbox)

        self.statusbar = QStatusBar(PolyTrend)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.showMessage("Ready to use PolyTrend")
        PolyTrend.setStatusBar(self.statusbar)

        PolyTrend.setCentralWidget(self.centralwidget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = PolyTrendUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
