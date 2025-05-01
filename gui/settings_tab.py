from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFileDialog, QSlider, QGroupBox, QFormLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QCheckBox, QSplitter, 
                             QListWidget, QSizePolicy, QListWidgetItem,  QAbstractItemView)
from PyQt6.QtCore import Qt
import os
import re
from utils.ui_components import ClickFilter, CustomLineEdit
from utils.dataset_manager import DatasetManager
from gui.class_colors_manager import ClassColorsManager
from gui.reorderable_sliders import ReorderableSliders

class AugmentationSettingsTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.skip_augmentations = {
            'Zoom': [],
            'Crop': [],
            'Rotate': [],
            'Mirror': [],
            'Overlay': []
        }

        self.class_colors = {}
        self.id_to_label = {}
        
        # Initialize class colors manager
        self.class_colors_manager = ClassColorsManager(self)
        
        # Default augmentation settings
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
        
        self.is_cancelled = False
        self.start_time = 0
        self.last_time_update = 0
        
        self.initUI()
        self.installEventFilter(ClickFilter(self))
        
    def initUI(self):
        layout = QVBoxLayout()

        # Directory selection
        dir_group = QGroupBox("Select Directories")
        dir_layout = QFormLayout()
        self.dataset_label = QLabel("Not selected")
        self.overlay_label = QLabel("Not selected")
        self.output_dir_label = QLabel("Not selected")
        self.dataset_btn = QPushButton("Select Dataset Root")
        self.overlay_btn = QPushButton("Select Overlay Image Directory")
        self.output_dir_btn = QPushButton("Select Output Directory")
        self.dataset_btn.clicked.connect(self.select_dataset_root)
        self.overlay_btn.clicked.connect(self.select_overlay_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        dir_layout.addRow(self.dataset_btn, self.dataset_label)
        dir_layout.addRow(self.overlay_btn, self.overlay_label)
        dir_layout.addRow(self.output_dir_btn, self.output_dir_label)
        dir_group.setLayout(dir_layout)
        dir_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(dir_group)

        # Sliders and Skip Augmentations
        weights_skip_layout = QSplitter(Qt.Orientation.Horizontal)
        
        # Weights sliders
        weights_group = QGroupBox("Augmentation Settings")
        weights_layout = QVBoxLayout()

        # Initialize reorderable sliders component
        self.reorderable_sliders = ReorderableSliders(self)
        
        # Add sliders to the component
        self.slider_data = [
            {"name": "Mirror % Probability:", "object": "mirror_slider", "value_object": "mirror_value", "aug_type": "mirror"},
            {"name": "Rotate % Probability:", "object": "rotate_slider", "value_object": "rotate_value", "aug_type": "rotate"},
            {"name": "Rotation (0 to 360) vs 90 %: ", "object": "rotation_random_vs_90_slider", "value_object": "rotation_random_vs_90_value", "aug_type": "rotation_random_vs_90"},
            {"name": "Crop % Probability:", "object": "crop_slider", "value_object": "crop_value", "aug_type": "crop"},
            {"name": "Maintain Aspect Ratio on Crop %: ", "object": "maintain_aspect_ratio_slider", "value_object": "maintain_aspect_ratio_value", "aug_type": "maintain_aspect_ratio"},
            {"name": "Zoom % Probability:", "object": "zoom_slider", "value_object": "zoom_value", "aug_type": "zoom"},
            {"name": "Zoom In vs Out %: ", "object": "zoom_in_vs_out_slider", "value_object": "zoom_in_vs_out_value", "aug_type": "zoom_in_vs_out"},
            {"name": "Overlay % Probability:", "object": "overlay_slider", "value_object": "overlay_value", "aug_type": "overlay"}
        ]

        for slider_info in self.slider_data:
            slider, value_edit = self.reorderable_sliders.add_slider(
                slider_info["name"], 
                slider_info["object"], 
                slider_info["value_object"],
                default_value=50,
                augmentation_type=slider_info["aug_type"]
            )
            
            # Store references for backward compatibility
            setattr(self, slider_info["object"], slider)
            setattr(self, slider_info["value_object"], value_edit)
        
        weights_layout.addWidget(self.reorderable_sliders)
        
        # Settings buttons group
        settings_buttons_layout = QHBoxLayout()
        self.skip_existing_checkbox = QCheckBox("Skip Already Augmented Images")
        self.skip_existing_checkbox.setChecked(True)
        settings_buttons_layout.addWidget(self.skip_existing_checkbox)
        
        self.load_config_button = QPushButton("Load Config")
        self.save_config_button = QPushButton("Save Config")
        
        self.load_config_button.clicked.connect(self.load_existing_config)
        self.save_config_button.clicked.connect(self.save_current_config)
        
        settings_buttons_layout.addWidget(self.load_config_button)
        settings_buttons_layout.addWidget(self.save_config_button)
        
        weights_layout.addLayout(settings_buttons_layout)
        weights_group.setLayout(weights_layout)

        # Skip Augmentations
        skip_colors_layout = QSplitter(Qt.Orientation.Vertical)

        self.skip_group = QGroupBox("Skip Augmentations for Folders")
        self.skip_layout = QVBoxLayout()
        self.skip_table = QTableWidget()
        self.skip_table.setColumnCount(7)  
        self.skip_table.setHorizontalHeaderLabels(['Folder', 'Zoom', 'Crop', 'Rotate', 'Mirror', 'Overlay', 'Skip All'])
        self.skip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.skip_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for col in range(1, 7):
            self.skip_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.skip_table.setColumnWidth(col, 50)
        self.skip_layout.addWidget(self.skip_table)
        self.skip_group.setLayout(self.skip_layout)
        
        skip_colors_layout.addWidget(self.skip_group)

        # Class Colors Group - Use the class colors manager widget
        self.class_color_group = QGroupBox("Class Colors")
        self.class_colors_layout = QVBoxLayout()
        
        # Store reference to the class colors table for compatibility
        self.class_colors_table = self.class_colors_manager.class_colors_table
        
        self.class_colors_layout.addWidget(self.class_colors_manager)
        self.class_color_group.setLayout(self.class_colors_layout)
        
        skip_colors_layout.addWidget(self.class_color_group)
        skip_colors_layout.setCollapsible(0, False)
        skip_colors_layout.setCollapsible(1, False)

        weights_skip_layout.addWidget(weights_group)
        weights_skip_layout.addWidget(skip_colors_layout)
        weights_skip_layout.setSizes([800, 400])  # Initial sizes of the panels
        weights_skip_layout.setCollapsible(0, False)
        weights_skip_layout.setCollapsible(1, False)

        # Set minimum sizes
        self.skip_group.setMinimumWidth(400)
        weights_skip_layout.setMinimumWidth(1200)
        weights_skip_layout.setHandleWidth(10)

        layout.addWidget(weights_skip_layout)
        self.setLayout(layout)

    def select_dataset_root(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if dir_name:
            self.dataset_label.setText(dir_name)
            self.parent.select_dataset_root(dir_name)

    def select_overlay_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Overlay Image Directory")
        if dir_name:
            self.overlay_label.setText(dir_name)
            self.parent.select_overlay_dir(dir_name)

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_dir_label.setText(dir_name)
            self.parent.select_output_dir(dir_name)

    def update_sliders_state(self):
        enable_normal_sliders = bool(self.parent.dataset_root)
        enable_overlay_sliders = bool(self.parent.overlay_image_dir)
        
        # Create filter lists for normal and overlay sliders
        normal_sliders = [
            'mirror_slider', 'crop_slider', 'zoom_slider', 'rotate_slider',
            'rotation_random_vs_90_slider', 'zoom_in_vs_out_slider', 'maintain_aspect_ratio_slider'
        ]
        overlay_sliders = ['overlay_slider']
        
        # Enable/disable normal sliders
        self.reorderable_sliders.enable_sliders(enable_normal_sliders, normal_sliders)
        
        # Enable/disable overlay sliders
        self.reorderable_sliders.enable_sliders(enable_overlay_sliders, overlay_sliders)

        # Update skip table checkboxes
        for row in range(self.skip_table.rowCount()):
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            if overlay_checkbox:
                overlay_checkbox.setEnabled(enable_overlay_sliders)

    def get_augmentation_order(self):
        """Get the current order of augmentations from the sliders list"""
        return self.reorderable_sliders.get_augmentation_order()

    def get_skip_augmentations(self):
        skip_augmentations = {
            'Zoom': [],
            'Crop': [],
            'Rotate': [],
            'Mirror': [],
            'Overlay': []
        }
        
        # Map column indices to augmentation types
        col_to_aug = {
            1: 'Zoom',
            2: 'Crop',
            3: 'Rotate',
            4: 'Mirror',
            5: 'Overlay'
        }
        
        # Check each row in the skip table
        for row in range(self.skip_table.rowCount()):
            folder_name = self.skip_table.item(row, 0).text()
            skip_all = self.skip_table.cellWidget(row, 6).isChecked()
            
            if skip_all:
                # If "Skip All" is checked, add folder to all augmentation types
                for aug_type in skip_augmentations:
                    skip_augmentations[aug_type].append(folder_name)
            else:
                # Check individual augmentation checkboxes
                for col, aug_type in col_to_aug.items():
                    if self.skip_table.cellWidget(row, col).isChecked():
                        skip_augmentations[aug_type].append(folder_name)
        
        return skip_augmentations

    def get_augmentation_params(self):
        # Get the current augmentation order
        augmentation_order = self.get_augmentation_order()
        
        # Get all parameters needed for augmentation
        params = {
            'skip_existing': self.skip_existing_checkbox.isChecked(),
            'skip_augmentations': self.get_skip_augmentations(),
            'mirror_weights': [self.reorderable_sliders.sliders.get('mirror_slider', 50).value(), 
                            100 - self.reorderable_sliders.sliders.get('mirror_slider', 50).value()],
            'crop_weights': [self.reorderable_sliders.sliders.get('crop_slider', 50).value(), 
                        100 - self.reorderable_sliders.sliders.get('crop_slider', 50).value()],
            'zoom_weights': [self.reorderable_sliders.sliders.get('zoom_slider', 50).value(), 
                        100 - self.reorderable_sliders.sliders.get('zoom_slider', 50).value()],
            'rotate_weights': [self.reorderable_sliders.sliders.get('rotate_slider', 50).value(), 
                            100 - self.reorderable_sliders.sliders.get('rotate_slider', 50).value()],
            'overlay_weights': ([self.reorderable_sliders.sliders.get('overlay_slider', 50).value(), 
                            100 - self.reorderable_sliders.sliders.get('overlay_slider', 50).value()] 
                            if self.parent.overlay_image_dir else [0, 100]),
            'rotation_random_vs_90_weights': [self.reorderable_sliders.sliders.get('rotation_random_vs_90_slider', 25).value(), 
                                            100 - self.reorderable_sliders.sliders.get('rotation_random_vs_90_slider', 25).value()],
            'overlay_min_max_scale': self.overlay_min_max_scale,
            'maintain_aspect_ratio_weights': [self.reorderable_sliders.sliders.get('maintain_aspect_ratio_slider', 50).value(),
                                        100 - self.reorderable_sliders.sliders.get('maintain_aspect_ratio_slider', 50).value()],
            'zoom_in_vs_out_weights': [self.reorderable_sliders.sliders.get('zoom_in_vs_out_slider', 40).value(),
                                    100 - self.reorderable_sliders.sliders.get('zoom_in_vs_out_slider', 40).value()],
            'zoom_padding': self.zoom_padding,
            'augmentation_order': augmentation_order
        }
        
        return params

    def update_class_colors_table(self):
        # Delegate to the class colors manager
        self.class_colors_manager.update_class_colors_table(self.class_colors, self.id_to_label)
    
    def toggle_skip_all(self, state, row):
        skip_all_checked = state == Qt.CheckState.Checked
        for col in range(1, 6):  # Update to check relevant columns
            checkbox = self.skip_table.cellWidget(row, col)
            checkbox.setEnabled(not skip_all_checked)
        if not self.parent.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            overlay_checkbox.setEnabled(False)

    def save_current_config(self):
        # Get the current order of sliders
        augmentation_order = self.get_augmentation_order()
        
        config_data = {
            "augmentation_order": augmentation_order,
            "crop_probability": self.crop_slider.value(),
            "maintain_aspect_ratio": self.maintain_aspect_ratio_slider.value(),
            "mirror_probability": self.mirror_slider.value(),
            "overlay_probability": self.overlay_slider.value(),
            "rotate_probability": self.rotate_slider.value(),
            "rotation_random_vs_90": self.rotation_random_vs_90_slider.value(),
            "zoom_in_vs_out": self.zoom_in_vs_out_slider.value(),
            "zoom_probability": self.zoom_slider.value(),
            "skip_existing": self.skip_existing_checkbox.isChecked()
        }
        self.parent.config_manager.save_config(config_data)

    def load_existing_config(self):
        config_data = self.parent.config_manager.load_config()
        if config_data:
            # Set slider values
            slider_values = {
                "crop_slider": config_data.get("crop_probability", 0),
                "maintain_aspect_ratio_slider": config_data.get("maintain_aspect_ratio", 0),
                "mirror_slider": config_data.get("mirror_probability", 0),
                "overlay_slider": config_data.get("overlay_probability", 0),
                "rotate_slider": config_data.get("rotate_probability", 0),
                "rotation_random_vs_90_slider": config_data.get("rotation_random_vs_90", 0),
                "zoom_in_vs_out_slider": config_data.get("zoom_in_vs_out", 0),
                "zoom_slider": config_data.get("zoom_probability", 0)
            }
            self.reorderable_sliders.set_slider_values(slider_values)
            
            # Set skip existing checkbox
            self.skip_existing_checkbox.setChecked(config_data.get("skip_existing", False))
            
            # Apply saved order if available
            if "augmentation_order" in config_data:
                self.reorderable_sliders.reorder_sliders_from_config(config_data["augmentation_order"])
            
            self.update_sliders_state()

    def scan_folders(self):
        dataset_root = self.parent.dataset_root
        
        # Clear previous data structures
        for key in self.skip_augmentations.keys():
            self.skip_augmentations[key] = []
        
        # Clear class mappings
        self.id_to_label.clear()
        self.class_colors.clear()

        # Scan dataset for folders and images
        folders = set()
        image_paths = []  # Reset image paths
        label_paths = {}  # Reset label paths

        for root, dirs, files in os.walk(dataset_root):
            if os.path.basename(root).lower() not in ['images', 'labels']:
                for name in dirs:
                    if name.lower() not in ['train', 'val', 'labels', 'images']:
                        folders.add(name)
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        image_paths.append(image_path)
                        label_path = os.path.join(dataset_root, 'labels', os.path.relpath(image_path, os.path.join(dataset_root, 'images')).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                        label_paths[image_path] = label_path

        folders = list(folders)
        folders.sort()

        self.skip_table.setRowCount(len(folders))

        # Keep a reference to the folder list in the image viewer
        if hasattr(self.parent, 'image_viewer_tab'):
            self.parent.image_viewer_tab.folder_list.clear()
            
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make folder names read-only
            self.skip_table.setItem(row, 0, folder_item)
            
            if hasattr(self.parent, 'image_viewer_tab'):
                list_item = QListWidgetItem(folder)
                self.parent.image_viewer_tab.folder_list.addItem(list_item)
                
            for col in range(1, 7):  # Update the range to include the new column
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 0px; margin-right: auto;")  # Align checkbox to the left 
                if col == 5:
                    checkbox.setEnabled(False)
                if col == 6:  # Connect the new checkbox to the slot
                    checkbox.stateChanged.connect(lambda state, r=row: self.toggle_skip_all(state, r))
                self.skip_table.setCellWidget(row, col, checkbox)

        # Sort images numerically
        image_paths.sort(key=self.natural_keys)
        
        # Update the parent's image paths
        self.parent.image_paths = image_paths
        self.parent.label_paths = label_paths

        # Parse YAML labels if available
        yaml_labels = DatasetManager.parse_dataset_yaml(dataset_root)
        
        # Update the stats tab with the labels
        if hasattr(self.parent, 'stats_tab'):
            self.parent.stats_tab.yaml_labels = yaml_labels

    def atoi(self, text):
        """Helper function for natural sort order"""
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        """Sort text strings in natural order"""
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]