# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QLabel,
    QMainWindow, QPlainTextEdit, QPushButton, QSizePolicy,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget)

class Ui_PolyTrend(object):
    def setupUi(self, PolyTrend):
        if not PolyTrend.objectName():
            PolyTrend.setObjectName(u"PolyTrend")
        PolyTrend.resize(530, 462)
        palette = QPalette()
        brush = QBrush(QColor(206, 206, 206, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(255, 109, 12, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        brush2 = QBrush(QColor(53, 53, 53, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Dark, brush2)
        brush3 = QBrush(QColor(226, 226, 226, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Text, brush3)
        brush4 = QBrush(QColor(0, 0, 0, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush4)
        brush5 = QBrush(QColor(88, 88, 88, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Window, brush5)
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Dark, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush3)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush4)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush5)
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Dark, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush5)
        PolyTrend.setPalette(palette)
        self.centralwidget = QWidget(PolyTrend)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.heading = QLabel(self.centralwidget)
        self.heading.setObjectName(u"heading")

        self.verticalLayout.addWidget(self.heading)

        self.intro = QLabel(self.centralwidget)
        self.intro.setObjectName(u"intro")

        self.verticalLayout.addWidget(self.intro)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.y_box = QPlainTextEdit(self.centralwidget)
        self.y_box.setObjectName(u"y_box")
        self.y_box.setMinimumSize(QSize(61, 121))
        self.y_box.setStyleSheet(u"color: black;")

        self.gridLayout.addWidget(self.y_box, 0, 3, 1, 1)

        self.degree_box = QTextEdit(self.centralwidget)
        self.degree_box.setObjectName(u"degree_box")
        self.degree_box.setMinimumSize(QSize(41, 121))
        self.degree_box.setStyleSheet(u"color: black;")

        self.gridLayout.addWidget(self.degree_box, 0, 5, 1, 1)

        self.x_box = QPlainTextEdit(self.centralwidget)
        self.x_box.setObjectName(u"x_box")
        self.x_box.setMinimumSize(QSize(61, 121))
        self.x_box.setStyleSheet(u"color: black;")

        self.gridLayout.addWidget(self.x_box, 0, 1, 1, 1)

        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)

        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)

        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 0, 6, 1, 1)

        self.extrap_box = QTextEdit(self.centralwidget)
        self.extrap_box.setObjectName(u"extrap_box")
        self.extrap_box.setStyleSheet(u"color: black;")

        self.gridLayout.addWidget(self.extrap_box, 0, 7, 1, 1)

        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 0, 4, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.csv_button = QPushButton(self.centralwidget)
        self.csv_button.setObjectName(u"csv_button")

        self.verticalLayout.addWidget(self.csv_button)

        self.plot = QPushButton(self.centralwidget)
        self.plot.setObjectName(u"plot")
        self.plot.setEnabled(False)

        self.verticalLayout.addWidget(self.plot)

        self.save_checkbox = QCheckBox(self.centralwidget)
        self.save_checkbox.setObjectName(u"save_checkbox")

        self.verticalLayout.addWidget(self.save_checkbox)

        PolyTrend.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(PolyTrend)
        self.statusbar.setObjectName(u"statusbar")
        PolyTrend.setStatusBar(self.statusbar)

        self.retranslateUi(PolyTrend)

        QMetaObject.connectSlotsByName(PolyTrend)
    # setupUi

    def retranslateUi(self, PolyTrend):
        PolyTrend.setWindowTitle(QCoreApplication.translate("PolyTrend", u"PolyTrend", None))
        self.heading.setText(QCoreApplication.translate("PolyTrend", u"This is PolyTrend - a simple clean way to create polynomial fits in data", None))
        self.intro.setText(QCoreApplication.translate("PolyTrend", u"For csv data, your data MUST be in the form x,f(x).\n"
"It is assumed that values on the same row correspond to the same point.\n"
"Make sure your CSV data starts with the axis lables in the first row\n"
"Accepted delimiters are whitespace and linebreaks in the text boxes below\n"
"Saved PNGs will be in the 'images' folder of the application\n"
"For the algebraic expressions of the models, visit the 'log' folder", None))
        self.y_box.setPlaceholderText(QCoreApplication.translate("PolyTrend", u"Input your y values...", None))
        self.degree_box.setPlaceholderText(QCoreApplication.translate("PolyTrend", u"Specify input degrees...", None))
        self.x_box.setPlaceholderText(QCoreApplication.translate("PolyTrend", u"Input your x values...", None))
        self.label_3.setText(QCoreApplication.translate("PolyTrend", u"x\n"
"values", None))
        self.label_4.setText(QCoreApplication.translate("PolyTrend", u"f(x)\n"
"values", None))
        self.label_6.setText(QCoreApplication.translate("PolyTrend", u"forecasts\n"
"(optional)", None))
        self.extrap_box.setPlaceholderText(QCoreApplication.translate("PolyTrend", u"Input data for model predictions...", None))
        self.label_5.setText(QCoreApplication.translate("PolyTrend", u"degrees", None))
        self.csv_button.setText(QCoreApplication.translate("PolyTrend", u"Import csv file", None))
        self.plot.setText(QCoreApplication.translate("PolyTrend", u"Fit polynomial", None))
        self.save_checkbox.setText(QCoreApplication.translate("PolyTrend", u"Save to PNG", None))
    # retranslateUi

