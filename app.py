import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import AugmentationGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AugmentationGUI()
    ex.show()
    sys.exit(app.exec())