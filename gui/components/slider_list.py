from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QListWidget, QListWidgetItem, QWidget,
                             QAbstractItemView, QSizePolicy)
from PyQt6.QtCore import Qt
from utils.ui_components import CustomLineEdit

class AugmentationSliderList(QGroupBox):
    """Component for managing augmentation sliders in a draggable list"""
    
    def __init__(self, parent):
        super().__init__("Augmentation Settings")
        self.parent = parent
        
        # Default weights - can be overridden by parent
        self.rotation_random_vs_90 = [25, 75]
        self.zoom_in_vs_out_weights = [40, 60]
        self.zoom_in_min_padding = 0.05
        self.zoom_in_max_padding = 0.5
        self.zoom_out_min_padding = 0.1
        self.zoom_out_max_padding = 0.8
        self.zoom_padding = [self.zoom_in_min_padding, self.zoom_in_max_padding, 
                            self.zoom_out_min_padding, self.zoom_out_max_padding]
        self.maintain_aspect_ratio_weights = [50, 50]
        self.overlay_min_max_scale = [0.3, 1.0]
        
        # Map slider attributes to augmentation types
        self.slider_to_augmentation_type = {
            "mirror_slider": "mirror",
            "rotate_slider": "rotate",
            "rotation_random_vs_90_slider": "rotation_random_vs_90",
            "crop_slider": "crop",
            "maintain_aspect_ratio_slider": "maintain_aspect_ratio",
            "zoom_slider": "zoom",
            "zoom_in_vs_out_slider": "zoom_in_vs_out",
            "overlay_slider": "overlay"
        }
        
        self.slider_data = [
            {"name": "Mirror % Probability:", "object": "mirror_slider", "value_object": "mirror_value"},
            {"name": "Rotate % Probability:", "object": "rotate_slider", "value_object": "rotate_value"},
            {"name": "Rotation (0 to 360) vs 90 %: ", "object": "rotation_random_vs_90_slider", "value_object": "rotation_random_vs_90_value"},
            {"name": "Crop % Probability:", "object": "crop_slider", "value_object": "crop_value"},
            {"name": "Maintain Aspect Ratio on Crop %: ", "object": "maintain_aspect_ratio_slider", "value_object": "maintain_aspect_ratio_value"},
            {"name": "Zoom % Probability:", "object": "zoom_slider", "value_object": "zoom_value"},
            {"name": "Zoom In vs Out %: ", "object": "zoom_in_vs_out_slider", "value_object": "zoom_in_vs_out_value"},
            {"name": "Overlay % Probability:", "object": "overlay_slider", "value_object": "overlay_value"}
        ]
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Create the list widget for sliders
        self.sliders_list = QListWidget()
        self.sliders_list.setDragEnabled(True)
        self.sliders_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.sliders_list.setMinimumHeight(300) 
        
        # Apply style to make selection transparent
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
        
        # Add sliders to the list
        for slider_info in self.slider_data:
            self.add_slider_to_list(slider_info["name"], slider_info["object"], slider_info["value_object"])
            
        layout.addWidget(self.sliders_list)
        self.setLayout(layout)
        
    def add_slider_to_list(self, name, slider_attr, value_attr):
        """Add a slider to the list widget"""
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
        slider, value_edit = self.create_slider()
        
        # Store references to these widgets
        setattr(self, slider_attr, slider)
        setattr(self, value_attr, value_edit)
        
        # If we already have values for this slider, use them
        if hasattr(self, slider_attr) and isinstance(getattr(self, slider_attr), QSlider):
            old_slider = getattr(self, slider_attr)
            if old_slider and hasattr(old_slider, 'value'):
                # Try to get the value of the previous slider
                try:
                    slider.setValue(old_slider.value())
                    value_edit.setText(str(old_slider.value()))
                except Exception:
                    # If it fails, just use default values
                    pass
        
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

    def create_slider(self):
        """Create a slider with a value label"""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setEnabled(False)
        
        value_label = CustomLineEdit("50")
        value_label.setFixedWidth(40)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setEnabled(False)
        
        # Connect signals
        slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
        value_label.textChanged.connect(lambda text, sld=slider: sld.setValue(int(text)) if text.isdigit() else None)
        
        return slider, value_label
    
    def update_sliders_state(self, dataset_root, overlay_image_dir):
        """Enable/disable sliders based on selected directories"""
        enable_normal_sliders = bool(dataset_root)
        enable_overlay_sliders = bool(overlay_image_dir)
        
        # Enable/disable normal sliders
        self.mirror_slider.setEnabled(enable_normal_sliders)
        self.mirror_value.setEnabled(enable_normal_sliders)
        self.crop_slider.setEnabled(enable_normal_sliders)
        self.crop_value.setEnabled(enable_normal_sliders)
        self.zoom_slider.setEnabled(enable_normal_sliders)
        self.zoom_value.setEnabled(enable_normal_sliders)
        self.rotate_slider.setEnabled(enable_normal_sliders)
        self.rotate_value.setEnabled(enable_normal_sliders)
        self.rotation_random_vs_90_slider.setEnabled(enable_normal_sliders)
        self.rotation_random_vs_90_value.setEnabled(enable_normal_sliders)
        self.zoom_in_vs_out_slider.setEnabled(enable_normal_sliders)
        self.zoom_in_vs_out_value.setEnabled(enable_normal_sliders)
        self.maintain_aspect_ratio_slider.setEnabled(enable_normal_sliders)
        self.maintain_aspect_ratio_value.setEnabled(enable_normal_sliders)
        
        # Enable/disable overlay sliders
        self.overlay_slider.setEnabled(enable_overlay_sliders)
        self.overlay_value.setEnabled(enable_overlay_sliders)
    
    def get_augmentation_order(self):
        """Get the current order of augmentations from the sliders list"""
        augmentation_order = []
        valid_augmentation_types = ["mirror", "crop", "zoom", "rotate", "overlay"]
        
        for i in range(self.sliders_list.count()):
            item_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
            for j in range(item_widget.layout().count()):
                widget = item_widget.layout().itemAt(j).widget()
                if isinstance(widget, QSlider):
                    for attr_name, attr_value in vars(self).items():
                        if attr_value is widget and attr_name in self.slider_to_augmentation_type:
                            aug_type = self.slider_to_augmentation_type[attr_name]
                            if aug_type in valid_augmentation_types:
                                augmentation_order.append(aug_type)
                            break
                    break
        
        return augmentation_order
    
    def get_augmentation_params(self):
        """Get all slider values as a dictionary"""
        return {
            'mirror_weights': [self.mirror_slider.value(), 100 - self.mirror_slider.value()],
            'crop_weights': [self.crop_slider.value(), 100 - self.crop_slider.value()],
            'zoom_weights': [self.zoom_slider.value(), 100 - self.zoom_slider.value()],
            'rotate_weights': [self.rotate_slider.value(), 100 - self.rotate_slider.value()],
            'overlay_weights': [self.overlay_slider.value(), 100 - self.overlay_slider.value()],
            'rotation_random_vs_90_weights': [self.rotation_random_vs_90_slider.value(), 
                                           100 - self.rotation_random_vs_90_slider.value()],
            'maintain_aspect_ratio_weights': [self.maintain_aspect_ratio_slider.value(),
                                          100 - self.maintain_aspect_ratio_slider.value()],
            'zoom_in_vs_out_weights': [self.zoom_in_vs_out_slider.value(),
                                    100 - self.zoom_in_vs_out_slider.value()],
            'zoom_padding': self.zoom_padding,
            'overlay_min_max_scale': self.overlay_min_max_scale,
            'augmentation_order': self.get_augmentation_order()
        }
        
    def reorder_sliders_from_config(self, order):
        """Reorder sliders based on the order saved in the config"""
        # Get current information about all sliders
        slider_info = []
        for i in range(self.sliders_list.count()):
            item = self.sliders_list.item(i)
            widget = self.sliders_list.itemWidget(item)
            
            # Find which slider this corresponds to
            for j in range(widget.layout().count()):
                child_widget = widget.layout().itemAt(j).widget()
                if isinstance(child_widget, QSlider):
                    # Find which attribute this slider corresponds to
                    for attr_name, attr_value in vars(self).items():
                        if attr_value is child_widget and attr_name in self.slider_to_augmentation_type:
                            aug_type = self.slider_to_augmentation_type[attr_name]
                            # Store current value
                            value = child_widget.value()
                            slider_info.append({
                                'aug_type': aug_type,
                                'slider_attr': attr_name,
                                'value': value,
                                'index': i
                            })
                            break
                    break
        
        # Create a mapping from augmentation types to their info
        aug_type_to_info = {info['aug_type']: info for info in slider_info}
        
        # Temporarily remove all sliders
        for i in range(self.sliders_list.count() - 1, -1, -1):
            self.sliders_list.takeItem(i)
        
        # Re-add sliders in the desired order
        for aug_type in order:
            if aug_type in aug_type_to_info:
                info = aug_type_to_info[aug_type]
                
                # Find the corresponding slider_info entry
                for slider_item in self.slider_data:
                    if slider_item['object'] == info['slider_attr']:
                        # Re-add the slider with proper values
                        self.add_slider_to_list(
                            slider_item["name"], 
                            slider_item["object"], 
                            slider_item["value_object"]
                        )
                        
                        # Set the value to match the original
                        slider = getattr(self, slider_item["object"])
                        slider.setValue(info['value'])
                        break