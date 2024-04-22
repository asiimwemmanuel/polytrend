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

import re
import sys
import shutil
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from polytrend import PolyTrend
from view.gui_ui import Ui_PolyTrend


class PolyTrendApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_PolyTrend()
        self.ui.setupUi(self)
        self.setWindowTitle("PolyTrend")

        self.ui.plot.clicked.connect(self.plot_graph)

        self.ui.csv_button.clicked.connect(self.import_csv)

        # Enable/disable the 'Find' button based on the data in the text boxes
        self.ui.x_box.textChanged.connect(self.manage_find_button)
        self.ui.y_box.textChanged.connect(self.manage_find_button)
        self.ui.degree_box.textChanged.connect(self.manage_find_button)

    def enable_csv_path_box(self):
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
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )

        if csv_path:
            df = pd.read_csv(csv_path)
            x_values = df[df.columns[0]].tolist()
            y_values = df[df.columns[1]].tolist()
            self.ui.x_box.setPlainText(" ".join(map(str, x_values)))
            self.ui.y_box.setPlainText(" ".join(map(str, y_values)))

            self.manage_find_button()

    def plot_graph(self):
        def _extract_values(text):
            if text.strip():
                values = re.split(r"[\s,\n]+", text.strip())
                return [float(x) for x in values if x]
            else:
                return []

        # Textual data processing & preparation for analysis
        x_values = _extract_values(self.ui.x_box.toPlainText())
        y_values = _extract_values(self.ui.y_box.toPlainText())
        extrap = _extract_values(self.ui.extrap_box.toPlainText())
        degrees = _extract_values(self.ui.degree_box.toPlainText())
        degrees = [int(x) for x in degrees]

        poly_trend = PolyTrend()

        poly_trend.polyplot(degrees, list((zip(x_values, y_values))), extrap)

    def closeEvent(self, event):
        folders_to_delete = ["./src/__pycache__", "./src/view/__pycache__"]
        for folder in folders_to_delete:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolyTrendApp()
    window.show()
    sys.exit(app.exec())
