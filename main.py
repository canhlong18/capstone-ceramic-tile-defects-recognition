"""Main program."""


from PyQt6.QtWidgets import QApplication
from MainWindow import MainWindow
import sys


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow(app)
    main_window.show()
    app.exec()
