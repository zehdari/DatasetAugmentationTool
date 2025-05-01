from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QAbstractItemView, QColorDialog, QSizePolicy)
from PyQt6.QtGui import QColor
import random

class ClassColorsManager(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.class_colors = {}
        self.id_to_label = {}
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        self.class_colors_table = QTableWidget()
        self.class_colors_table.setColumnCount(2)
        self.class_colors_table.setHorizontalHeaderLabels(['Class', 'Color'])
        self.class_colors_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_colors_table.itemClicked.connect(self.on_color_cell_clicked)
        self.class_colors_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        
        # Set size policy - using direct policy creation instead of referencing parent
        self.class_colors_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout.addWidget(self.class_colors_table)
        self.setLayout(layout)
        
    def on_color_cell_clicked(self, item):
        try:
            if item.column() == 1:  # Check if the clicked cell is in the color column
                row = item.row()
                class_label = self.class_colors_table.item(row, 0).text()
                
                # Find the class_id that corresponds to this label
                class_id = None
                for cid, label in self.id_to_label.items():
                    if str(label) == class_label:
                        class_id = cid
                        break
                
                # If we couldn't find the ID through the mapping, 
                # the label might be the raw class ID itself
                if class_id is None:
                    class_id = class_label
                    
                # Make sure the class_id exists in class_colors
                if class_id in self.class_colors:
                    # Store current table state before changing
                    current_colors = self.class_colors.copy()
                    current_id_to_label = self.id_to_label.copy()
                    
                    # Show color dialog
                    color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
                    if color.isValid():
                        self.class_colors[class_id] = color
                        
                        # Update just the color cell, not the entire table
                        color_item = QTableWidgetItem()
                        color_item.setBackground(color)
                        self.class_colors_table.setItem(row, 1, color_item)
                        
                        # Update the image in the viewer tab if available
                        if hasattr(self.parent.parent, 'image_viewer_tab'):
                            self.parent.parent.image_viewer_tab.show_image()
        except Exception as e:
            print(f"Error in on_color_cell_clicked: {str(e)}")
            pass
        
    def update_class_colors_table(self, class_colors=None, id_to_label=None):
        """Update the class colors table with current class data"""
        # Store current selection if any
        current_row = self.class_colors_table.currentRow()
        current_col = self.class_colors_table.currentColumn()

        if class_colors is not None:
            self.class_colors = class_colors
        if id_to_label is not None:
            self.id_to_label = id_to_label

        # Clear and reset table
        self.class_colors_table.setRowCount(len(self.class_colors))
        row = 0
        for class_id, color in self.class_colors.items():
            # Get label for class ID, defaulting to class_id itself if not found
            class_label = self.id_to_label.get(class_id, class_id)
            
            class_item = QTableWidgetItem(str(class_label))
            color_item = QTableWidgetItem()
            color_item.setBackground(color)
            self.class_colors_table.setItem(row, 0, class_item)
            self.class_colors_table.setItem(row, 1, color_item)
            row += 1
            
        # Restore selection if possible
        if current_row >= 0 and current_row < self.class_colors_table.rowCount():
            self.class_colors_table.setCurrentCell(current_row, current_col)
            
    def generate_color_for_class(self, class_id):
        """Generate and save a random color for a class if it doesn't exist"""
        if class_id not in self.class_colors:
            self.class_colors[class_id] = QColor(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.class_colors[class_id]