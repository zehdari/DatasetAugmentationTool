import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QSlider, QLineEdit, QTreeWidget, QTreeWidgetItem, 
                            QDoubleSpinBox, QFormLayout)
from PyQt6.QtCore import Qt, QModelIndex
from PyQt6.QtGui import QDropEvent


class ReorderableTreeWidget(QTreeWidget):
    """Tree widget with drag-drop support for top-level (parent) nodes only"""
    def __init__(self, headers=None, parent=None):
        super().__init__(parent)
        
        # Setup drag-drop settings
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        
        # Setup appearance
        if headers:
            self.setHeaderLabels(headers)
        else:
            self.setHeaderLabels(["Item", "Value"])
        self.setColumnCount(2)
        self.header().setStretchLastSection(True)
        
        # Disable selection highlighting
        self.setStyleSheet("""
            QTreeWidget::item:selected { 
                background: transparent; 
                color: black;
            }
            QTreeWidget::item:hover { 
                background: transparent; 
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        """
        Strictly enforce drag-drop rules:
        1. Only top-level items can be dragged
        2. Items can only be dropped at the top level
        3. Items cannot become children of other items
        """
        # Debug output
        print("--- Drop Event Started ---")
        
        if not self.selectedItems():
            print("DEBUG: No selected items, ignoring drop")
            event.ignore()
            return
            
        selected_item = self.selectedItems()[0]
        print(f"DEBUG: Selected item text: '{selected_item.text(0)}'")
        
        # Only allow top-level items to be dragged
        if selected_item.parent() is not None:
            print("DEBUG: Not a top-level item, ignoring drop")
            event.ignore()
            return
        
        # Determine drop position
        drop_pos = event.position().toPoint()
        drop_index = self.indexAt(drop_pos)
        
        print(f"DEBUG: Drop position: {drop_pos.x()}, {drop_pos.y()}")
        print(f"DEBUG: Drop index valid: {drop_index.isValid()}")
        
        # Prevent dropping onto items (which would make them children)
        if drop_index.isValid():
            item_at_pos = self.itemFromIndex(drop_index)
            print(f"DEBUG: Drop indicator position: {self.dropIndicatorPosition()}")
            
            # If dropping directly on an item or into a child item position
            if item_at_pos and self.dropIndicatorPosition() == QTreeWidget.DropIndicatorPosition.OnItem:
                print("DEBUG: Dropping onto item, ignoring")
                event.ignore()
                return
            
            # If trying to drop as a child of another item
            if drop_index.parent().isValid():
                print("DEBUG: Trying to drop as child, ignoring")
                event.ignore()
                return
            
            if item_at_pos:
                print(f"DEBUG: Item at position: '{item_at_pos.text(0)}'")
        
        # Explicitly handle moving to specific positions in the root
        # to make sure we're only reordering at the top level
        from_index = self.indexOfTopLevelItem(selected_item)
        print(f"DEBUG: Moving from index: {from_index}")
        
        # Determine the target index where the item will be moved
        indicator_pos = self.dropIndicatorPosition()
        drop_item = self.itemAt(drop_pos)
        
        if drop_item:
            # If drop_item is a child, find its top-level parent
            orig_drop_item = drop_item
            while drop_item.parent():
                drop_item = drop_item.parent()
            
            if orig_drop_item != drop_item:
                print(f"DEBUG: Original drop item '{orig_drop_item.text(0)}' is a child, using parent '{drop_item.text(0)}'")
                
            to_index = self.indexOfTopLevelItem(drop_item)
            print(f"DEBUG: Drop item: '{drop_item.text(0)}', To index: {to_index}")
            
            if indicator_pos == QTreeWidget.DropIndicatorPosition.AboveItem:
                # Move above target
                print(f"DEBUG: Drop above, target index remains: {to_index}")
            elif indicator_pos == QTreeWidget.DropIndicatorPosition.BelowItem:
                # Move below target
                to_index = to_index + 1
                print(f"DEBUG: Drop below, target index becomes: {to_index}")
            else:
                # Don't allow dropping onto items (making them children)
                print("DEBUG: Not above or below, ignoring")
                event.ignore()
                return
        else:
            # If dropping at the end of the list
            to_index = self.topLevelItemCount()
            print(f"DEBUG: Dropping at end of list, to index: {to_index}")
        
        # Adjust index if moving down since we'll remove the item first
        if from_index < to_index:
            to_index -= 1
            print(f"DEBUG: Adjusted to_index for downward movement: {to_index}")
        
        # Skip if not actually moving
        if from_index == to_index:
            print("DEBUG: Not actually moving (same index), accepting event")
            event.accept()
            return
        
        print(f"DEBUG: Before move: topLevelItemCount = {self.topLevelItemCount()}")
        
        # CRITICAL FIX: Block signals during the move to prevent widget deletion
        self.blockSignals(True)
        
        # Create a clone of the item to preserve its structure
        # We'll manually transfer all data and children
        clone_item = QTreeWidgetItem()
        clone_item.setText(0, selected_item.text(0))
        clone_item.setText(1, selected_item.text(1))
        clone_item.setFlags(selected_item.flags())
        
        print(f"DEBUG: Created clone of '{selected_item.text(0)}'")
        
        # Store all widgets for the top-level item
        widgets = {}
        for column in range(self.columnCount()):
            widget = self.itemWidget(selected_item, column)
            if widget:
                widgets[column] = widget
                print(f"DEBUG: Stored widget for column {column}")
        
        # Function to recursively clone a tree item with all its children
        def clone_with_children(source_item, target_item):
            # Clone all children
            for i in range(source_item.childCount()):
                child = source_item.child(i)
                child_clone = QTreeWidgetItem()
                child_clone.setText(0, child.text(0))
                child_clone.setText(1, child.text(1))
                child_clone.setFlags(child.flags())
                target_item.addChild(child_clone)
                
                # Store widgets for this child
                for column in range(self.columnCount()):
                    widget = self.itemWidget(child, column)
                    if widget:
                        print(f"DEBUG: Stored widget for '{child.text(0)}' column {column}")
                        # We'll restore this later
                        self.removeItemWidget(child, column)
                        self.setItemWidget(child_clone, column, widget)
                
                # Recursively handle grandchildren
                clone_with_children(child, child_clone)
        
        # Clone the entire hierarchy
        clone_with_children(selected_item, clone_item)
        
        # Remove the original item - we won't use it again
        self.takeTopLevelItem(from_index)
        print(f"DEBUG: Removed original item from index {from_index}")
        
        # Insert the clone at the new position
        self.insertTopLevelItem(to_index, clone_item)
        print(f"DEBUG: Inserted clone at index {to_index}")
        
        # Restore widgets for the top-level item
        for column, widget in widgets.items():
            print(f"DEBUG: Restoring widget for column {column}")
            self.setItemWidget(clone_item, column, widget)
        
        # Unblock signals
        self.blockSignals(False)
        
        # Select the moved item
        self.setCurrentItem(clone_item)
        clone_item.setExpanded(True)  # Make sure it's expanded
        
        print(f"DEBUG: After move: topLevelItemCount = {self.topLevelItemCount()}")
        print("--- Drop Event Finished ---")
        
        event.accept()


class ReorderableTreeManager(QWidget):
    """Generalizable widget to manage a reorderable tree"""
    def __init__(self, headers=None, parent=None):
        super().__init__(parent)
        
        # Initialize with optional custom headers
        self.headers = headers if headers else ["Item", "Value"]
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create tree widget
        self.tree = ReorderableTreeWidget(self.headers)
        layout.addWidget(self.tree)
        
        # No save/load buttons as requested
        self.setLayout(layout)
    
    def addParentItem(self, text, flags=None):
        """Add a parent item with special drag-drop handlers"""
        if flags is None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | \
                   Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
                   
        item = QTreeWidgetItem(self.tree, [text])
        item.setFlags(flags)
        
        # Add handle indicator for visual clarity
        item.setText(0, f"≡ {text}")
        
        return item
    
    def addChildItem(self, parent, text, flags=None):
        """Add a child item (not draggable)"""
        if flags is None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            
        item = QTreeWidgetItem(parent, [text])
        item.setFlags(flags)
        
        return item
    
    def addSliderToItem(self, item, param_name, initial_value, min_val=0, max_val=100):
        """Add a slider widget to a tree item"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Create slider and value display
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(initial_value)
        
        value_edit = QLineEdit(str(initial_value))
        value_edit.setMaximumWidth(40)
        
        # Connect them
        slider.valueChanged.connect(lambda val, edit=value_edit: edit.setText(str(val)))
        value_edit.textChanged.connect(
            lambda text, sld=slider: sld.setValue(int(text)) if text.isdigit() else None
        )
        
        # Add label
        label = QLabel(f"{param_name.title()}:")
        label.setMinimumWidth(80)
        
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value_edit)
        
        self.tree.setItemWidget(item, 1, widget)
        return widget
    
    def addDoubleSpinBoxToItem(self, item, value, min_val, max_val, label_text="Value:", decimals=2):
        """Add a double spin box to a tree item"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.setSingleStep(0.01)
        spinbox.setDecimals(decimals)
        
        label = QLabel(label_text)
        label.setMinimumWidth(80)
        
        layout.addWidget(label)
        layout.addWidget(spinbox)
        
        self.tree.setItemWidget(item, 1, widget)
        return widget
    
    def addFormWidgetToItem(self, item, widget_pairs):
        """Add a form layout with multiple widgets to an item"""
        container = QWidget()
        form_layout = QFormLayout(container)
        form_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add all widget pairs (label, widget)
        for label_text, widget in widget_pairs:
            form_layout.addRow(label_text, widget)
        
        self.tree.setItemWidget(item, 1, container)
        return container
    
    def getTopLevelItemOrder(self):
        """Get the current order of top-level items"""
        order = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            name = item.text(0).replace('≡ ', '')
            order.append(name)
        return order
    
    def expandAll(self):
        """Expand all tree items"""
        self.tree.expandAll()
    
    def getTreeWidget(self):
        """Return the tree widget for direct access if needed"""
        return self.tree


# Example usage
class AugmentationTree(ReorderableTreeManager):
    """Specific implementation for augmentation settings"""
    def __init__(self):
        super().__init__(headers=["Augmentation", "Parameters"])
        self.setupAugmentationTree()
    
    def setupAugmentationTree(self):
        """Set up the augmentation-specific tree structure"""
        # Create all parent nodes
        zoom_parent = self.addParentItem("Zoom")
        rotate_parent = self.addParentItem("Rotate")
        mirror_parent = self.addParentItem("Mirror")
        crop_parent = self.addParentItem("Crop")
        overlay_parent = self.addParentItem("Overlay")
        
        # Add parameters to parents
        self.addSliderToItem(zoom_parent, "probability", 50)
        self.addSliderToItem(rotate_parent, "probability", 50)
        self.addSliderToItem(mirror_parent, "probability", 50)
        self.addSliderToItem(crop_parent, "probability", 50)
        self.addSliderToItem(overlay_parent, "probability", 30)
        
        # Add children
        # Zoom children
        zoom_in = self.addChildItem(zoom_parent, "Zoom In")
        self.addSliderToItem(zoom_in, "probability", 40)
        
        zoom_out = self.addChildItem(zoom_parent, "Zoom Out")
        self.addSliderToItem(zoom_out, "probability", 60)
        
        # Add min/max padding to zoom in/out
        zoom_in_min = self.addChildItem(zoom_in, "Min Padding")
        self.addDoubleSpinBoxToItem(zoom_in_min, 0.05, 0.01, 0.3)
        
        zoom_in_max = self.addChildItem(zoom_in, "Max Padding")
        self.addDoubleSpinBoxToItem(zoom_in_max, 0.5, 0.1, 0.7)
        
        zoom_out_min = self.addChildItem(zoom_out, "Min Padding")
        self.addDoubleSpinBoxToItem(zoom_out_min, 0.1, 0.05, 0.5)
        
        zoom_out_max = self.addChildItem(zoom_out, "Max Padding")
        self.addDoubleSpinBoxToItem(zoom_out_max, 0.8, 0.2, 0.95)
        
        # Rotate children
        rotate_random = self.addChildItem(rotate_parent, "Random Angle")
        self.addSliderToItem(rotate_random, "probability", 25)
        
        rotate_90 = self.addChildItem(rotate_parent, "90° Increments")
        self.addSliderToItem(rotate_90, "probability", 75)
        
        # Crop child
        maintain_aspect = self.addChildItem(crop_parent, "Maintain Aspect Ratio")
        self.addSliderToItem(maintain_aspect, "probability", 50)
        
        # Overlay child with form layout
        overlay_scale = self.addChildItem(overlay_parent, "Scale")
        
        # Create spinboxes for min/max scale
        min_scale = QDoubleSpinBox()
        min_scale.setRange(0.1, 0.9)
        min_scale.setValue(0.3)
        min_scale.setSingleStep(0.1)
        
        max_scale = QDoubleSpinBox()
        max_scale.setRange(0.2, 2.0)
        max_scale.setValue(1.0)
        max_scale.setSingleStep(0.1)
        
        # Add both to a form layout
        self.addFormWidgetToItem(overlay_scale, [
            ("Min Scale:", min_scale),
            ("Max Scale:", max_scale)
        ])
        
        # Expand all items
        self.expandAll()
    
    def getZoomPaddingArray(self):
        """Get the zoom padding array in the expected format"""
        padding = [0.05, 0.5, 0.1, 0.8]  # Default values
        tree = self.getTreeWidget()
        
        # Find the zoom parent
        for i in range(tree.topLevelItemCount()):
            parent = tree.topLevelItem(i)
            if "Zoom" in parent.text(0):
                # Find zoom in/out children
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    
                    if "Zoom In" in child.text(0):
                        # Find min/max padding
                        for k in range(child.childCount()):
                            padding_item = child.child(k)
                            widget = tree.itemWidget(padding_item, 1)
                            if widget:
                                spinbox = widget.findChild(QDoubleSpinBox)
                                if spinbox:
                                    if "Min" in padding_item.text(0):
                                        padding[0] = spinbox.value()
                                    elif "Max" in padding_item.text(0):
                                        padding[1] = spinbox.value()
                    
                    elif "Zoom Out" in child.text(0):
                        # Find min/max padding
                        for k in range(child.childCount()):
                            padding_item = child.child(k)
                            widget = tree.itemWidget(padding_item, 1)
                            if widget:
                                spinbox = widget.findChild(QDoubleSpinBox)
                                if spinbox:
                                    if "Min" in padding_item.text(0):
                                        padding[2] = spinbox.value()
                                    elif "Max" in padding_item.text(0):
                                        padding[3] = spinbox.value()
        
        return padding


def main():
    app = QApplication(sys.argv)
    window = AugmentationTree()
    window.setWindowTitle("Augmentation Settings")
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()