from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QListWidget, QListWidgetItem, QSizePolicy,
                             QAbstractItemView, QPushButton, QTextEdit, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap
from utils.ui_components import CustomLineEdit


class CollapsibleDetails(QWidget):
    """A widget that can be expanded/collapsed to show/hide details."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        self.collapsed = True
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(0)  # Reduce vertical spacing
        
        # Button for expanding/collapsing
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        
        self.toggle_btn = QPushButton("▼ Show Details")
        self.toggle_btn.setFlat(True)  # Make button more compact
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                text-align: left;
                padding: 0px;
                margin: 0px;
                font-size: 10px;  /* Smaller font */
                color: #555;
            }
            QPushButton:hover {
                color: #4682B4;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_details)
        self.toggle_btn.setFixedHeight(16)  # Reduce button height
        header_layout.addWidget(self.toggle_btn)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Details content (hidden by default)
        self.details_edit = QTextEdit()
        self.details_edit.setPlaceholderText("Enter details about this augmentation...")
        self.details_edit.setFixedHeight(60)  # Fixed compact height
        self.details_edit.setVisible(False)
        self.details_edit.setStyleSheet("""
            QTextEdit {
                font-size: 10px;
                padding: 2px;
                background-color: transparent;  /* Changed from #f8f8f8 to transparent */
            }
        """)
        
        # Add a horizontal line above the details
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.separator.setVisible(False)
        self.separator.setFixedHeight(1)  # Make separator very thin
        
        layout.addWidget(self.separator)
        layout.addWidget(self.details_edit)
        
    def toggle_details(self):
        self.collapsed = not self.collapsed
        self.details_edit.setVisible(not self.collapsed)
        self.separator.setVisible(not self.collapsed)
        
        # Update button text
        if self.collapsed:
            self.toggle_btn.setText("▼ Show Details")
        else:
            self.toggle_btn.setText("▲ Hide Details")
            
    def set_details_text(self, text):
        self.details_edit.setText(text)
        
    def get_details_text(self):
        return self.details_edit.toPlainText()
        
    def set_enabled(self, enabled):
        self.toggle_btn.setEnabled(enabled)
        self.details_edit.setEnabled(enabled)
        
class ReorderableSliders(QWidget):
    """A widget that displays a list of sliders that can be reordered by drag and drop."""
    
    # Signal emitted when any slider value changes
    sliderValueChanged = pyqtSignal(str, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.sliders = {}  # Dictionary to store slider references
        self.values = {}   # Dictionary to store value edit references
        self.details = {}  # Dictionary to store details widgets
        self.slider_to_augmentation_type = {}  # Mapping from slider names to augmentation types
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # Reduce spacing between list items
        
        self.sliders_list = QListWidget()
        self.sliders_list.setDragEnabled(True)
        self.sliders_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.sliders_list.setMinimumHeight(250)  # Reduced from 300
        self.sliders_list.setSpacing(0)  # Minimal spacing between items
        
        self.sliders_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
            }
            QListWidget::item {
                padding-top: 0px;
                padding-bottom: 0px;
                margin-top: 0px;
                margin-bottom: 0px;
            }
            QListWidget::item:selected { 
                background: transparent; 
                color: black;
            }
            QListWidget::item:hover { 
                background: transparent; 
                border: none;
            }
            QListWidget::item:selected:active {
                background: transparent;
                color: black;
            }
            QListWidget::item:selected:!active {
                background: transparent;
                color: black;
            }
        """)
        
        layout.addWidget(self.sliders_list)
        self.setLayout(layout)
        
    def add_slider(self, name, slider_attr, value_attr, default_value=50, augmentation_type=None, details_text=""):
        """Add a new slider to the list with the given name and attributes."""
        # Create a main widget to hold both slider and details
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(0)  # Reduce vertical spacing between slider and details
        
        # Create a widget to hold the slider row
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(5, 3, 5, 3)  # Reduced vertical margins (3px instead of 5px)
        
        # Create drag handle label
        drag_handle = QLabel("≡")  # Using equal sign to create a simple handle icon
        drag_handle.setFixedWidth(15)  # Reduced width
        drag_handle.setStyleSheet("""
            font-size: 16px;  /* Slightly smaller font */
            color: #999;
            padding: 0px;
            margin: 0px;
            border-radius: 2px;
        """)
        
        # Create label
        label = QLabel(name)
        label.setMinimumWidth(180)  # Same width as before
        label.setStyleSheet("padding: 0px; margin: 0px;")  # Remove padding
        
        # Create slider and value
        slider, value_edit = self.create_slider(default_value)
        
        # Store references to these widgets
        self.sliders[slider_attr] = slider
        self.values[value_attr] = value_edit
        
        if augmentation_type:
            self.slider_to_augmentation_type[slider_attr] = augmentation_type
        
        # Add to slider layout
        slider_layout.addWidget(drag_handle)
        slider_layout.addWidget(label)
        slider_layout.addWidget(slider, 1)  # Give slider stretch factor
        slider_layout.addWidget(value_edit)
        
        # Create details section
        details_widget = CollapsibleDetails()
        details_widget.set_details_text(details_text)
        self.details[slider_attr] = details_widget
        
        # Set fixed heights for more compact layout
        slider_widget.setFixedHeight(30)  # Compact height for slider row
        
        # Add widgets to main layout
        main_layout.addWidget(slider_widget)
        main_layout.addWidget(details_widget)
        
        # Add row to list widget
        list_item = QListWidgetItem(self.sliders_list)
        
        # Calculate appropriate size hint
        collapsed_height = slider_widget.sizeHint().height() + details_widget.sizeHint().height()
        expanded_height = collapsed_height + details_widget.details_edit.height() + 5  # Add a little padding
        
        # Set initial size for collapsed state (we'll resize when expanded)
        list_item.setSizeHint(QSize(main_widget.sizeHint().width(), collapsed_height))
        
        # Add the item to the list and set the widget
        self.sliders_list.addItem(list_item)
        self.sliders_list.setItemWidget(list_item, main_widget)
        
        # Connect toggle signal to resize the list item
        details_widget.toggle_btn.clicked.connect(
            lambda checked=None, item=list_item, widget=details_widget, 
            slider_h=slider_widget.sizeHint().height(), 
            details_h=details_widget.details_edit.height():
            self._adjust_item_size(item, widget.collapsed, slider_h, details_h))
        
        return slider, value_edit, details_widget
        
    def _adjust_item_size(self, item, is_collapsed, slider_height, details_height):
        """Adjust the size of a list item based on collapse state"""
        if is_collapsed:  # Details were just hidden
            new_height = slider_height + 16  # Just the slider + small space for toggle button
        else:  # Details were just shown
            new_height = slider_height + details_height + 20  # Add space for details
        
        item.setSizeHint(QSize(item.sizeHint().width(), new_height))

    def create_slider(self, default_value=50):
        """Create a new slider with a value label."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(default_value)
        slider.setEnabled(False)
        slider.setFixedHeight(32)  # Reduce slider height
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px;
                margin: 0px;
            }
            QSlider::handle:horizontal {
                width: 10px;
                margin: -3px 0px;
            }
        """)
        
        value_label = CustomLineEdit(str(default_value))
        value_label.setFixedWidth(35)  # Slightly smaller width
        value_label.setFixedHeight(18)  # Consistent height with slider
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setEnabled(False)
        value_label.setStyleSheet("padding: 0px; margin: 0px;")
        
        # Connect signals
        slider.valueChanged.connect(
            lambda value, lbl=value_label: lbl.setText(str(value))
        )
        slider.valueChanged.connect(
            lambda value, sld=slider: self._on_slider_value_changed(sld, value)
        )
        value_label.textChanged.connect(
            lambda text, sld=slider: sld.setValue(int(text)) if text.isdigit() else None
        )
        
        return slider, value_label
    
    def _on_slider_value_changed(self, slider, value):
        """Handle slider value changes and emit signals."""
        # Find the slider attribute name
        for attr_name, s in self.sliders.items():
            if s is slider:
                self.sliderValueChanged.emit(attr_name, value)
                break
    
    def get_augmentation_order(self):
        """Get the current order of augmentations from the sliders list."""
        augmentation_order = []
        valid_augmentation_types = ["mirror", "crop", "zoom", "rotate", "overlay"]
        
        for i in range(self.sliders_list.count()):
            main_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
            slider_widget = main_widget.layout().itemAt(0).widget()
            
            for j in range(slider_widget.layout().count()):
                widget = slider_widget.layout().itemAt(j).widget()
                if isinstance(widget, QSlider):
                    for attr_name, attr_value in self.sliders.items():
                        if attr_value is widget and attr_name in self.slider_to_augmentation_type:
                            aug_type = self.slider_to_augmentation_type[attr_name]
                            if aug_type in valid_augmentation_types:
                                augmentation_order.append(aug_type)
                            break
                    break
        
        return augmentation_order
    
    def get_details_texts(self):
        """Get a dictionary of all details texts, keyed by slider attribute names."""
        details_texts = {}
        for attr_name, details_widget in self.details.items():
            details_texts[attr_name] = details_widget.get_details_text()
        return details_texts
    
    def set_slider_values(self, values_dict):
        """Set multiple slider values from a dictionary.
        
        Args:
            values_dict (dict): Dictionary mapping slider attribute names to values
        """
        for slider_name, value in values_dict.items():
            if slider_name in self.sliders:
                # Set the slider value and ensure it's within valid range
                value = max(0, min(100, int(value)))  # Ensure value is in range 0-100
                
                try:
                    # Set slider value and update the corresponding text field
                    slider = self.sliders[slider_name]
                    slider.setValue(value)
                    
                    # Also update the corresponding value edit if it exists
                    value_attr = slider_name.replace('slider', 'value')
                    if value_attr in self.values:
                        self.values[value_attr].setText(str(value))
                except Exception as e:
                    print(f"Error setting value for slider {slider_name}: {e}")
    
    def set_details_values(self, details_dict):
        """Set multiple details texts from a dictionary.
        
        Args:
            details_dict (dict): Dictionary mapping slider attribute names to details texts
        """
        for slider_name, text in details_dict.items():
            if slider_name in self.details:
                try:
                    details_widget = self.details[slider_name]
                    details_widget.set_details_text(text)
                except Exception as e:
                    print(f"Error setting details for slider {slider_name}: {e}")
    
    def enable_sliders(self, enable=True, filter_list=None):
        """Enable or disable sliders, optionally filtering by a list of names."""
        for name, slider in self.sliders.items():
            if filter_list is None or name in filter_list:
                slider.setEnabled(enable)
                # Also enable/disable the corresponding value edit
                for value_name, value_edit in self.values.items():
                    if value_name == name.replace('slider', 'value'):
                        value_edit.setEnabled(enable)
                        break
                # Also enable/disable the corresponding details widget
                if name in self.details:
                    self.details[name].set_enabled(enable)
    
    def reorder_sliders_from_config(self, order):
        """Reorder sliders based on a list of augmentation types."""
        # Create a mapping from augmentation types to slider attributes
        aug_to_slider = {}
        for slider_attr, aug_type in self.slider_to_augmentation_type.items():
            aug_to_slider[aug_type] = slider_attr
        
        # Store slider configs keyed by slider attribute
        slider_configs = {}
        for slider_attr, slider in self.sliders.items():
            value = slider.value()
            enabled = slider.isEnabled()
            
            # Find corresponding value edit
            value_attr = slider_attr.replace('slider', 'value')
            value_edit = self.values.get(value_attr)
            value_edit_enabled = value_edit.isEnabled() if value_edit else False
            
            # Get details widget and its text
            details_widget = self.details.get(slider_attr)
            details_text = details_widget.get_details_text() if details_widget else ""
            
            # Find the label for this slider
            label_text = None
            for i in range(self.sliders_list.count()):
                main_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
                slider_widget = main_widget.layout().itemAt(0).widget()
                for j in range(slider_widget.layout().count()):
                    widget = slider_widget.layout().itemAt(j).widget()
                    if isinstance(widget, QSlider) and widget is slider:
                        # Found the right widget, now look for the label in the same layout
                        for k in range(slider_widget.layout().count()):
                            label_widget = slider_widget.layout().itemAt(k).widget()
                            if isinstance(label_widget, QLabel) and label_widget.width() > 20:  # Likely the label (not drag handle)
                                label_text = label_widget.text()
                                break
                        break
                if label_text:
                    break
                    
            # If we couldn't find a label, use a generic one
            if not label_text:
                aug_type = self.slider_to_augmentation_type.get(slider_attr, "")
                label_text = f"{aug_type.capitalize() if aug_type else 'Unknown'} Slider"
                
            # Store the config
            slider_configs[slider_attr] = {
                'value': value,
                'enabled': enabled,
                'value_edit_enabled': value_edit_enabled,
                'label': label_text,
                'details_text': details_text,
                'aug_type': self.slider_to_augmentation_type.get(slider_attr)
            }
        
        # Remember the mapping of augmentation types
        old_mapping = self.slider_to_augmentation_type.copy()
        
        # Temporarily remove all sliders from display
        self.sliders_list.clear()
        
        # Clear internal references but keep values
        old_sliders = self.sliders.copy()
        old_values = self.values.copy()
        old_details = self.details.copy()
        
        self.sliders.clear()
        self.values.clear()
        self.details.clear()
        self.slider_to_augmentation_type.clear()
        
        # Re-add sliders in the desired order
        for aug_type in order:
            if aug_type in aug_to_slider:
                slider_attr = aug_to_slider[aug_type]
                value_attr = slider_attr.replace('slider', 'value')
                
                if slider_attr in slider_configs:
                    config = slider_configs[slider_attr]
                    
                    # Add the slider with the original label, value and details
                    slider, value_edit, details_widget = self.add_slider(
                        name=config['label'],
                        slider_attr=slider_attr,
                        value_attr=value_attr,
                        default_value=config['value'],
                        augmentation_type=config['aug_type'],
                        details_text=config['details_text']
                    )
                    
                    # Restore enabled state
                    slider.setEnabled(config['enabled'])
                    if value_edit:
                        value_edit.setEnabled(config['value_edit_enabled'])
                    if details_widget:
                        details_widget.set_enabled(config['enabled'])
        
        # Add any sliders that weren't in the order but were in the original set
        for slider_attr, config in slider_configs.items():
            if slider_attr not in self.sliders:
                value_attr = slider_attr.replace('slider', 'value')
                
                # Add the slider with the original label, value and details
                slider, value_edit, details_widget = self.add_slider(
                    name=config['label'],
                    slider_attr=slider_attr,
                    value_attr=value_attr,
                    default_value=config['value'],
                    augmentation_type=config['aug_type'],
                    details_text=config['details_text']
                )
                
                # Restore enabled state
                slider.setEnabled(config['enabled'])
                if value_edit:
                    value_edit.setEnabled(config['value_edit_enabled'])
                if details_widget:
                    details_widget.set_enabled(config['enabled'])
                    
    def clear(self):
        """Clear all sliders from the list."""
        self.sliders_list.clear()
        self.sliders.clear()
        self.values.clear()
        self.details.clear()
        self.slider_to_augmentation_type.clear()