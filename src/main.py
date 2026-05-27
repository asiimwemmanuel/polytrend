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

    self.ui.x_box.textChanged.connect(self.toggle_find_button_state)
    self.ui.y_box.textChanged.connect(self.toggle_find_button_state)
    self.ui.error_box.textChanged.connect(self.toggle_find_button_state)
    self.ui.degree_box.textChanged.connect(self.toggle_find_button_state)

  def toggle_find_button_state(self):
    x_vals = self.ui.x_box.toPlainText().split()
    y_vals = self.ui.y_box.toPlainText().split()
    err_vals = self.ui.error_box.toPlainText().split()
    degrees = self.ui.degree_box.toPlainText().split()

    enabled = all([
      len(x_vals) == len(y_vals),
      len(x_vals) >= 2,
      len(degrees) >= 1,
      len(err_vals) in {0, 1, len(y_vals)},
    ])
    self.ui.plot.setEnabled(enabled)

  def import_csv(self):
    csv_path, _ = QFileDialog.getOpenFileName(
      self, "Open CSV File", "", "CSV Files (*.csv)"
    )
    if not csv_path:
      return

    x_vals, y_vals, err_vals = [], [], []

    with open(csv_path, newline="") as csvfile:
      csvreader = reader(csvfile)
      next(csvreader)  # skip header
      for row in csvreader:
        x_vals.append(row[0])
        y_vals.append(row[1])
        err_vals.append(row[2] if len(row) >= 3 else 0)

    self.ui.x_box.setPlainText("\n".join(map(str, x_vals)))
    self.ui.y_box.setPlainText("\n".join(map(str, y_vals)))
    self.ui.error_box.setPlainText("\n".join(map(str, err_vals)))
    self.toggle_find_button_state()

  def plot_graph(self):
    def _parse(text) -> list[float]:
      if not text.strip():
        return []
      return [float(v) for v in split(r"[\s,\n]+", text.strip()) if v]

    x_vals = _parse(self.ui.x_box.toPlainText())
    y_vals = _parse(self.ui.y_box.toPlainText())
    err_vals = _parse(self.ui.error_box.toPlainText()) or [0.0]
    # If a single error value is given, apply it to all points
    err_vals.extend([err_vals[0]] * (len(y_vals) - len(err_vals)))

    extrap = _parse(self.ui.extrap_box.toPlainText())
    degrees = [int(v) for v in _parse(self.ui.degree_box.toPlainText())]

    PolyTrend().polyplot(degrees, list(zip(x_vals, y_vals, err_vals)), extrap or None)

  def closeEvent(self, event):
    base_dir = path.dirname(path.abspath(__file__))
    cache_dirs = [
      path.join(base_dir, "__pycache__"),
      path.join(base_dir, "view/__pycache__"),
    ]
    for folder in cache_dirs:
      try:
        rmtree(folder)
        print(f"Deleted '{folder}'.")
      except FileNotFoundError:
        pass  # nothing to clean up
      except PermissionError:
        print(f"Permission denied: '{folder}'.")
      except OSError as e:
        print(f"Error deleting '{folder}': {e}")
    event.accept()


if __name__ == "__main__":
  app = QApplication(argv)
  window = PolyTrendApp()
  window.show()
  exit(app.exec())
