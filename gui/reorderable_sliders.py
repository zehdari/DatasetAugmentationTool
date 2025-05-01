from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QListWidget, QListWidgetItem, QSizePolicy,
                             QAbstractItemView)
from PyQt6.QtCore import Qt, pyqtSignal
from utils.ui_components import CustomLineEdit


class ReorderableSliders(QWidget):
    """A widget that displays a list of sliders that can be reordered by drag and drop."""
    
    # Signal emitted when any slider value changes
    sliderValueChanged = pyqtSignal(str, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.sliders = {}  # Dictionary to store slider references
        self.values = {}   # Dictionary to store value edit references
        self.slider_to_augmentation_type = {}  # Mapping from slider names to augmentation types
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.sliders_list = QListWidget()
        self.sliders_list.setDragEnabled(True)
        self.sliders_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.sliders_list.setMinimumHeight(300) 
        
        self.sliders_list.setStyleSheet("""
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
        
    def add_slider(self, name, slider_attr, value_attr, default_value=50, augmentation_type=None):
        """Add a new slider to the list with the given name and attributes."""
        # Create a widget to hold the slider row
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create drag handle label
        drag_handle = QLabel("≡")  # Using equal sign to create a simple handle icon
        drag_handle.setFixedWidth(20)
        drag_handle.setStyleSheet("""
            font-size: 18px; 
            color: #999;
            padding: 2px;
            border-radius: 3px;
        """)
        
        # Create label
        label = QLabel(name)
        label.setMinimumWidth(180)  # Reduced width slightly to make room for handle
        
        # Create slider and value
        slider, value_edit = self.create_slider(default_value)
        
        # Store references to these widgets
        self.sliders[slider_attr] = slider
        self.values[value_attr] = value_edit
        
        if augmentation_type:
            self.slider_to_augmentation_type[slider_attr] = augmentation_type
        
        # Add to layout
        item_layout.addWidget(drag_handle)
        item_layout.addWidget(label)
        item_layout.addWidget(slider, 1)  # Give slider stretch factor
        item_layout.addWidget(value_edit)
        
        # Add row to list widget
        list_item = QListWidgetItem(self.sliders_list)
        list_item.setSizeHint(item_widget.sizeHint())
        self.sliders_list.addItem(list_item)
        self.sliders_list.setItemWidget(list_item, item_widget)
        
        return slider, value_edit

    def create_slider(self, default_value=50):
        """Create a new slider with a value label."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(default_value)
        slider.setEnabled(False)
        
        value_label = CustomLineEdit(str(default_value))
        value_label.setFixedWidth(40)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setEnabled(False)
        
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
            item_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
            for j in range(item_widget.layout().count()):
                widget = item_widget.layout().itemAt(j).widget()
                if isinstance(widget, QSlider):
                    for attr_name, attr_value in self.sliders.items():
                        if attr_value is widget and attr_name in self.slider_to_augmentation_type:
                            aug_type = self.slider_to_augmentation_type[attr_name]
                            if aug_type in valid_augmentation_types:
                                augmentation_order.append(aug_type)
                            break
                    break
        
        return augmentation_order
    
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
            
            # Find the label for this slider
            label_text = None
            for i in range(self.sliders_list.count()):
                item_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
                for j in range(item_widget.layout().count()):
                    widget = item_widget.layout().itemAt(j).widget()
                    if isinstance(widget, QSlider) and widget is slider:
                        # Found the right widget, now look for the label in the same layout
                        for k in range(item_widget.layout().count()):
                            label_widget = item_widget.layout().itemAt(k).widget()
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
                'aug_type': self.slider_to_augmentation_type.get(slider_attr)
            }
        
        # Remember the mapping of augmentation types
        old_mapping = self.slider_to_augmentation_type.copy()
        
        # Temporarily remove all sliders from display
        self.sliders_list.clear()
        
        # Clear internal references but keep values
        old_sliders = self.sliders.copy()
        old_values = self.values.copy()
        
        self.sliders.clear()
        self.values.clear()
        self.slider_to_augmentation_type.clear()
        
        # Re-add sliders in the desired order
        for aug_type in order:
            if aug_type in aug_to_slider:
                slider_attr = aug_to_slider[aug_type]
                value_attr = slider_attr.replace('slider', 'value')
                
                if slider_attr in slider_configs:
                    config = slider_configs[slider_attr]
                    
                    # Add the slider with the original label and value
                    slider, value_edit = self.add_slider(
                        name=config['label'],
                        slider_attr=slider_attr,
                        value_attr=value_attr,
                        default_value=config['value'],
                        augmentation_type=config['aug_type']
                    )
                    
                    # Restore enabled state
                    slider.setEnabled(config['enabled'])
                    if value_edit:
                        value_edit.setEnabled(config['value_edit_enabled'])
        
        # Add any sliders that weren't in the order but were in the original set
        for slider_attr, config in slider_configs.items():
            if slider_attr not in self.sliders:
                value_attr = slider_attr.replace('slider', 'value')
                
                # Add the slider with the original label and value
                slider, value_edit = self.add_slider(
                    name=config['label'],
                    slider_attr=slider_attr,
                    value_attr=value_attr,
                    default_value=config['value'],
                    augmentation_type=config['aug_type']
                )
                
                # Restore enabled state
                slider.setEnabled(config['enabled'])
                if value_edit:
                    value_edit.setEnabled(config['value_edit_enabled'])
    def clear(self):
        """Clear all sliders from the list."""
        self.sliders_list.clear()
        self.sliders.clear()
        self.values.clear()
        self.slider_to_augmentation_type.clear()