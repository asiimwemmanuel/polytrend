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

from csv import reader
from os import path
from re import split
from shutil import rmtree
from sys import argv, exit

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

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
        self.ui.x_box.textChanged.connect(self.toggle_find_button_state)
        self.ui.y_box.textChanged.connect(self.toggle_find_button_state)
        self.ui.error_box.textChanged.connect(self.toggle_find_button_state)
        self.ui.degree_box.textChanged.connect(self.toggle_find_button_state)

    # def enable_csv_path_box(self):
    #     self.ui.save_checkbox.setEnabled(True)

    def toggle_find_button_state(self):
        x_values = self.ui.x_box.toPlainText().split()
        y_values = self.ui.y_box.toPlainText().split()
        err_values = self.ui.error_box.toPlainText().split()
        degrees = self.ui.degree_box.toPlainText().split()

        if all(
            [
                len(x_values) == len(y_values),
                len(x_values) >= 2,
                len(degrees) >= 1,
                len(err_values) in {0, 1, len(y_values)},
            ]
        ):
            self.ui.plot.setEnabled(True)
        else:
            self.ui.plot.setEnabled(False)

    def import_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )

        if csv_path:
            x_values = []
            y_values = []
            err_values = []

            with open(csv_path, newline="") as csvfile:
                # Create a CSV reader object
                csvreader = reader(csvfile)

                # Skip the header row
                next(csvreader)

                # Iterate over each row in the CSV file
                for row in csvreader:
                    # Append data from each column to respective lists
                    x_values.append(row[0])
                    y_values.append(row[1])
                    # Check if there is data in the third column
                    if len(row) == 3:
                        err_values.append(row[2])
                    else:
                        err_values.append(0)

            # * A note on Albert's old request
            # * If whitespace existed in production prior to the request, split() parameter changes in plot_graph() wouldn't be necessary
            self.ui.x_box.setPlainText("\n".join(map(str, x_values)))
            self.ui.y_box.setPlainText("\n".join(map(str, y_values)))
            self.ui.error_box.setPlainText("\n".join(map(str, err_values)))

            self.toggle_find_button_state()

    def plot_graph(self):
        def _extract_values(text) -> list[float]:
            if text.strip():
                values = split(r"[\s,\n]+", text.strip())
                return [float(x) for x in values if x]
            else:
                return []

        # Textual data processing & preparation for analysis
        x_values = _extract_values(self.ui.x_box.toPlainText())
        y_values = _extract_values(self.ui.y_box.toPlainText())
        err_values = _extract_values(self.ui.error_box.toPlainText())
        err_values = [0.00] if not err_values else err_values

        # Allows user to input only 1 value to apply to all the others
        # Minor efficiency gain: Directly replicate the first element of err_values if its length is 1 :)
        # * the gain only works if the user EITHER inputs 1 value OR provides values for each point individually
        err_values.extend([err_values[0]] * (len(y_values) - len(err_values)))

        extrap = _extract_values(self.ui.extrap_box.toPlainText())
        degrees = [int(x) for x in _extract_values(self.ui.degree_box.toPlainText())]

        poly_trend = PolyTrend()

        poly_trend.polyplot(
            degrees, list((zip(x_values, y_values, err_values))), extrap
        )

    def closeEvent(self, event):
        # Construct absolute paths for the folders to delete
        base_dir = path.dirname(path.abspath(__file__))
        folders_to_delete = [
            path.join(base_dir, "__pycache__"),
            path.join(base_dir, "view/__pycache__"),
        ]
        for folder in folders_to_delete:
            try:
                rmtree(folder)
                print(f"Folder '{folder}' deleted successfully.")
            except FileNotFoundError:
                raise FileNotFoundError(f"Folder '{folder}' not found.")
            except PermissionError:
                print(f"Permission denied to delete the folder '{folder}'.")
            except OSError as e:
                print(f"Error occurred while deleting folder '{folder}': {e}")

        event.accept()


if __name__ == "__main__":
    app = QApplication(argv)
    window = PolyTrendApp()
    window.show()
    exit(app.exec())
