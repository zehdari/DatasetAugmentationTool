from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtCore import QObject, QEvent

class ClickFilter(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            for widget in obj.findChildren(QLineEdit):
                widget.clearFocus()
        return super().eventFilter(obj, event)

class CustomLineEdit(QLineEdit):
    def focusOutEvent(self, event):
        self.deselect()
        super().focusOutEvent(event)