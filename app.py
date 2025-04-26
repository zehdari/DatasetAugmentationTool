import sys
from PyQt6.QtWidgets import QApplication
from augmentation_gui import AugmentionGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AugmentionGUI()
    ex.show()
    sys.exit(app.exec())