from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QColorDialog, QSizePolicy)
from PyQt6.QtGui import QColor

class ClassColorTable(QGroupBox):
    """Component for managing class colors"""
    
    def __init__(self, parent):
        super().__init__("Class Colors")
        self.parent = parent
        
        # Initialize data structures
        self.class_colors = {}
        self.id_to_label = {}
        self.label_to_id = {}
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Create table widget
        self.class_colors_table = QTableWidget()
        self.class_colors_table.setColumnCount(2)
        self.class_colors_table.setHorizontalHeaderLabels(['Class', 'Color'])
        self.class_colors_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_colors_table.itemClicked.connect(self.on_color_cell_clicked)
        self.class_colors_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.class_colors_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout.addWidget(self.class_colors_table)
        self.setLayout(layout)
        
    def on_color_cell_clicked(self, item):
        """Handle click on a color cell to change class color"""
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
                    color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
                    if color.isValid():
                        self.class_colors[class_id] = color
                        self.update_class_colors_table()
                        # Update the image in the viewer tab if available
                        if hasattr(self.parent.parent, 'image_viewer_tab'):
                            self.parent.parent.image_viewer_tab.show_image()
        except Exception as e:
            print(f"Error in on_color_cell_clicked: {str(e)}")
            pass
            
    def update_class_colors_table(self):
        """Update the table with current class colors"""
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
            
    def set_class_mappings(self, id_to_label, label_to_id):
        """Set the mappings between class IDs and labels"""
        self.id_to_label = id_to_label
        self.label_to_id = label_to_id
        
    def update_colors(self, yaml_labels=None):
        """Update the colors based on class labels and generate colors if needed"""
        # If we have label mappings, assign colors to each class
        class_ids = list(self.id_to_label.keys()) if self.id_to_label else []
        
        # If we have yaml labels but no mappings, create mappings
        if yaml_labels and not class_ids:
            for i, label in enumerate(yaml_labels):
                self.id_to_label[i] = label
                self.label_to_id[label] = i
                class_ids.append(i)
                
        # Generate colors for any classes that don't have them
        for class_id in class_ids:
            if class_id not in self.class_colors:
                # Generate a color using hue based on class index
                hue = (hash(str(class_id)) % 360) / 360.0  # Normalize to 0-1
                self.class_colors[class_id] = QColor.fromHslF(hue, 0.8, 0.5)
                
        # Update the table with the new colors
        self.update_class_colors_table()
        
    def get_class_colors(self):
        """Return the current class colors"""
        return self.class_colors