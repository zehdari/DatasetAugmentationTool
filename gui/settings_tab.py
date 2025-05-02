from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFileDialog, QSlider, QGroupBox, QFormLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QCheckBox, QSplitter, 
                             QListWidget, QSizePolicy, QListWidgetItem,  QAbstractItemView)
from PyQt6.QtCore import Qt
import os
import re
from utils.ui_components import ClickFilter, CustomLineEdit
from utils.dataset_manager import DatasetManager
from utils.augmentation_config_loader import AugmentationConfigLoader
from gui.class_colors_manager import ClassColorsManager
from gui.reorderable_sliders import ReorderableSliders

class AugmentationSettingsTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        # Load augmentation configuration from YAML
        self.config_loader = AugmentationConfigLoader('config/augmentation_config.yaml')
        self.config_loader.load_config()
        
        # Initialize skip augmentations dictionary from config
        self.skip_augmentations = {category: [] for category in self.config_loader.get_skip_categories()}

        self.class_colors = {}
        self.id_to_label = {}
        
        # Initialize class colors manager
        self.class_colors_manager = ClassColorsManager(self)
        
        # Default parameter values (will be overridden by loaded config if available)
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
        
        # Get configured augmentations, child sliders, and float params
        augmentations = self.config_loader.get_augmentations()
        child_sliders_config = self.config_loader.get_child_sliders()
        float_params_config = self.config_loader.get_float_params()
        
        # Process parent sliders based on configuration
        for aug_type, aug_info in augmentations.items():
            slider, value_edit, details_widget = self.reorderable_sliders.add_slider(
                name=aug_info["name"],
                slider_attr=aug_info["slider_attr"],
                value_attr=aug_info["value_attr"],
                default_value=aug_info.get("default_value", 50),
                augmentation_type=aug_type,
                details_text=aug_info.get("details", "")
            )
            
            # Store references for backward compatibility
            setattr(self, aug_info["slider_attr"], slider)
            setattr(self, aug_info["value_attr"], value_edit)
        
        # Process child sliders based on configuration
        for parent_attr, children in child_sliders_config.items():
            for child_info in children:
                child_attr = child_info["attr"]
                child_name = child_info["name"]
                child_default = child_info.get("default", 50)
                
                child_slider = self.reorderable_sliders.add_child_slider(
                    parent_attr=parent_attr,
                    name=child_name,
                    child_attr=child_attr,
                    default_value=child_default
                )
                
                # Store references for backward compatibility
                if child_slider:
                    setattr(self, child_attr, child_slider)
        
        # Process float inputs based on configuration
        for parent_attr, inputs in float_params_config.items():
            for input_info in inputs:
                input_attr = input_info["attr"]
                input_name = input_info["name"]
                default_value = input_info.get("default", 0.5)
                min_value = input_info.get("min", 0.0)
                max_value = input_info.get("max", 1.0)
                step = input_info.get("step", 0.01)
                
                float_input = self.reorderable_sliders.add_float_input(
                    parent_attr=parent_attr,
                    name=input_name,
                    input_attr=input_attr,
                    default_value=default_value,
                    min_value=min_value,
                    max_value=max_value,
                    step=step
                )
        
        # Connect signals
        self.reorderable_sliders.childSliderValueChanged.connect(self.on_child_slider_value_changed)
        self.reorderable_sliders.floatValueChanged.connect(self.on_float_value_changed)
        
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

        # Skip Augmentations - dynamically create based on config
        skip_colors_layout = QSplitter(Qt.Orientation.Vertical)

        self.skip_group = QGroupBox("Skip Augmentations for Folders")
        self.skip_layout = QVBoxLayout()
        self.skip_table = QTableWidget()
        
        # Set columns based on configured skip categories
        skip_categories = self.config_loader.get_skip_categories()
        self.skip_table.setColumnCount(len(skip_categories) + 2)  # +2 for Folder and Skip All columns
        
        # Set table headers
        headers = ['Folder'] + skip_categories + ['Skip All']
        self.skip_table.setHorizontalHeaderLabels(headers)
        
        self.skip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.skip_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        
        # Set column widths
        for col in range(1, len(skip_categories) + 2):
            self.skip_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.skip_table.setColumnWidth(col, 50)
        
        self.skip_layout.addWidget(self.skip_table)
        self.skip_group.setLayout(self.skip_layout)
        
        skip_colors_layout.addWidget(self.skip_group)

        # Class Colors Group - use the class colors manager widget
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

    def on_child_slider_value_changed(self, child_attr, value):
        """Handle child slider value changes."""
        # Update class attributes for backward compatibility
        if hasattr(self, child_attr) and child_attr == "rotation_random_vs_90_slider":
            # Update rotation_random_vs_90 weights
            self.rotation_random_vs_90 = [value, 100 - value]
        elif hasattr(self, child_attr) and child_attr == "zoom_in_vs_out_slider":
            # Update zoom_in_vs_out weights
            self.zoom_in_vs_out_weights = [value, 100 - value]
        elif hasattr(self, child_attr) and child_attr == "maintain_aspect_ratio_slider":
            # Update maintain_aspect_ratio weights
            self.maintain_aspect_ratio_weights = [value, 100 - value]
            
    def on_float_value_changed(self, input_attr, value):
        """Handle float input value changes."""
        # Update class attributes or data structures as needed
        if input_attr == "zoom_in_min_padding":
            self.zoom_in_min_padding = value
            self.zoom_padding[0] = value
        elif input_attr == "zoom_in_max_padding":
            self.zoom_in_max_padding = value
            self.zoom_padding[1] = value
        elif input_attr == "zoom_out_min_padding":
            self.zoom_out_min_padding = value
            self.zoom_padding[2] = value
        elif input_attr == "zoom_out_max_padding":
            self.zoom_out_max_padding = value
            self.zoom_padding[3] = value
        elif input_attr == "overlay_min_scale":
            self.overlay_min_max_scale[0] = value
        elif input_attr == "overlay_max_scale":
            self.overlay_min_max_scale[1] = value

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
        
        # Get all augmentation sliders from config
        augmentations = self.config_loader.get_augmentations()
        
        # Create lists for normal and overlay sliders
        normal_sliders = []
        overlay_sliders = []
        
        # Categorize sliders based on augmentation type
        for aug_type, aug_info in augmentations.items():
            slider_attr = aug_info["slider_attr"]
            if aug_type == "overlay":
                overlay_sliders.append(slider_attr)
            else:
                normal_sliders.append(slider_attr)
                
        # Add child sliders to normal sliders
        child_sliders_config = self.config_loader.get_child_sliders()
        for parent_attr, children in child_sliders_config.items():
            if parent_attr not in overlay_sliders:  # Only add children of normal sliders
                for child_info in children:
                    normal_sliders.append(child_info["attr"])
                    
        # Add float inputs to appropriate lists
        float_params_config = self.config_loader.get_float_params()
        for parent_attr, inputs in float_params_config.items():
            if parent_attr in overlay_sliders:
                for input_info in inputs:
                    overlay_sliders.append(input_info["attr"])
            else:
                for input_info in inputs:
                    normal_sliders.append(input_info["attr"])
        
        # Enable/disable normal sliders
        self.reorderable_sliders.enable_sliders(enable_normal_sliders, normal_sliders)
        
        # Enable/disable overlay sliders
        self.reorderable_sliders.enable_sliders(enable_overlay_sliders, overlay_sliders)

        # Update skip table checkboxes
        skip_categories = self.config_loader.get_skip_categories()
        overlay_col = next((i for i, cat in enumerate(skip_categories, 1) if cat == "Overlay"), None)
        
        if overlay_col:
            for row in range(self.skip_table.rowCount()):
                overlay_checkbox = self.skip_table.cellWidget(row, overlay_col)
                if overlay_checkbox:
                    overlay_checkbox.setEnabled(enable_overlay_sliders)

    def get_augmentation_order(self):
        """Get the current order of augmentations from the sliders list"""
        return self.reorderable_sliders.get_augmentation_order()

    def get_skip_augmentations(self):
        skip_categories = self.config_loader.get_skip_categories()
        skip_augmentations = {category: [] for category in skip_categories}
        
        # Map column indices to augmentation types
        col_to_aug = {idx+1: category for idx, category in enumerate(skip_categories)}
        
        # Check each row in the skip table
        for row in range(self.skip_table.rowCount()):
            folder_name = self.skip_table.item(row, 0).text()
            skip_all_col = len(skip_categories) + 1
            skip_all = self.skip_table.cellWidget(row, skip_all_col).isChecked()
            
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
        
        # Get slider values
        slider_values = self.reorderable_sliders.get_slider_values()
        
        # Get float input values
        float_values = self.reorderable_sliders.get_float_values()
        
        # Update zoom_padding and overlay_min_max_scale with current values from float inputs
        self.zoom_padding = [
            float_values.get('zoom_in_min_padding', self.zoom_in_min_padding),
            float_values.get('zoom_in_max_padding', self.zoom_in_max_padding),
            float_values.get('zoom_out_min_padding', self.zoom_out_min_padding),
            float_values.get('zoom_out_max_padding', self.zoom_out_max_padding)
        ]
        
        self.overlay_min_max_scale = [
            float_values.get('overlay_min_scale', self.overlay_min_max_scale[0]),
            float_values.get('overlay_max_scale', self.overlay_min_max_scale[1])
        ]
        
        # Initialize parameters dictionary with common parameters
        params = {
            'skip_existing': self.skip_existing_checkbox.isChecked(),
            'skip_augmentations': self.get_skip_augmentations(),
            'augmentation_order': augmentation_order,
            'zoom_padding': self.zoom_padding,
            'overlay_min_max_scale': self.overlay_min_max_scale
        }
        
        # Get configured augmentations
        augmentations = self.config_loader.get_augmentations()
        
        # Dynamically add parameters for all augmentation types
        for aug_type, aug_info in augmentations.items():
            slider_attr = aug_info["slider_attr"]
            value = slider_values.get(slider_attr, aug_info.get("default_value", 50))
            weight_param_name = f"{aug_type}_weights"
            
            # Special case for overlay - check if directory is selected
            if aug_type == "overlay" and not self.parent.overlay_image_dir:
                params[weight_param_name] = [0, 100]  # Disable overlay
            else:
                params[weight_param_name] = [value, 100 - value]
        
        # Add parameters for child sliders
        child_sliders_config = self.config_loader.get_child_sliders()
        for parent_attr, children in child_sliders_config.items():
            for child_info in children:
                child_attr = child_info["attr"]
                value = slider_values.get(child_attr, child_info.get("default", 50))
                
                # Map well-known child sliders to their parameter names
                if child_attr == "rotation_random_vs_90_slider":
                    params["rotation_random_vs_90_weights"] = [value, 100 - value]
                elif child_attr == "maintain_aspect_ratio_slider":
                    params["maintain_aspect_ratio_weights"] = [value, 100 - value]
                elif child_attr == "zoom_in_vs_out_slider":
                    params["zoom_in_vs_out_weights"] = [value, 100 - value]
                else:
                    # For custom child sliders, use a predictable naming pattern
                    weight_param_name = f"{child_attr.replace('_slider', '')}_weights"
                    params[weight_param_name] = [value, 100 - value]
        
        # Add all float parameters as is
        for attr, value in float_values.items():
            params[attr] = value
        
        return params
        
    def update_class_colors_table(self):
        # Delegate to the class colors manager
        self.class_colors_manager.update_class_colors_table(self.class_colors, self.id_to_label)
    
    def toggle_skip_all(self, state, row):
        skip_all_checked = state == Qt.CheckState.Checked
        skip_categories = self.config_loader.get_skip_categories()
        
        # Update each augmentation checkbox based on Skip All state
        for col in range(1, len(skip_categories) + 1):
            checkbox = self.skip_table.cellWidget(row, col)
            if checkbox:
                checkbox.setEnabled(not skip_all_checked)
        
        # Special handling for overlay if needed
        overlay_col = next((i for i, cat in enumerate(skip_categories, 1) if cat == "Overlay"), None)
        if overlay_col and not self.parent.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, overlay_col)
            if overlay_checkbox:
                overlay_checkbox.setEnabled(False)

    def save_current_config(self):
        # Get the current order of sliders
        augmentation_order = self.get_augmentation_order()
        
        # Get details text for all sliders
        details_texts = self.reorderable_sliders.get_details_texts()
        
        # Get all slider values
        slider_values = self.reorderable_sliders.get_slider_values()
        
        # Get all float input values
        float_values = self.reorderable_sliders.get_float_values()
        
        # Create config data dictionary
        config_data = {
            "augmentation_order": augmentation_order,
            "skip_existing": self.skip_existing_checkbox.isChecked(),
            "details_texts": details_texts
        }
        
        # Add all slider values
        for slider_attr, value in slider_values.items():
            # Convert slider_attr to a more readable config key
            config_key = slider_attr.replace("_slider", "_probability")
            config_data[config_key] = value
            
        # Add all float values
        for float_attr, value in float_values.items():
            config_data[float_attr] = value
            
        # Save to JSON file
        self.parent.config_manager.save_config(config_data)

    def load_existing_config(self):
        config_data = self.parent.config_manager.load_config()
        if config_data:
            # Get the known augmentations
            augmentations = self.config_loader.get_augmentations()
            
            # Set slider values for known augmentations
            slider_values = {}
            for aug_type, aug_info in augmentations.items():
                slider_attr = aug_info["slider_attr"]
                config_key = slider_attr.replace("_slider", "_probability")
                if config_key in config_data:
                    slider_values[slider_attr] = config_data[config_key]
                    
            # Set child slider values
            child_sliders_config = self.config_loader.get_child_sliders()
            for parent_attr, children in child_sliders_config.items():
                for child_info in children:
                    child_attr = child_info["attr"]
                    config_key = child_attr.replace("_slider", "")
                    if config_key in config_data:
                        slider_values[child_attr] = config_data[config_key]
            
            # Update all slider values
            self.reorderable_sliders.set_slider_values(slider_values)
            
            # Set float input values
            float_values = {}
            float_params_config = self.config_loader.get_float_params()
            for parent_attr, inputs in float_params_config.items():
                for input_info in inputs:
                    input_attr = input_info["attr"]
                    if input_attr in config_data:
                        float_values[input_attr] = config_data[input_attr]
            
            # Update all float input values
            self.reorderable_sliders.set_float_values(float_values)
            
            # Update class attributes to match loaded values
            if "zoom_in_min_padding" in float_values:
                self.zoom_in_min_padding = float_values["zoom_in_min_padding"]
            if "zoom_in_max_padding" in float_values:
                self.zoom_in_max_padding = float_values["zoom_in_max_padding"]
            if "zoom_out_min_padding" in float_values:
                self.zoom_out_min_padding = float_values["zoom_out_min_padding"]
            if "zoom_out_max_padding" in float_values:
                self.zoom_out_max_padding = float_values["zoom_out_max_padding"]
                
            self.zoom_padding = [
                self.zoom_in_min_padding,
                self.zoom_in_max_padding,
                self.zoom_out_min_padding,
                self.zoom_out_max_padding
            ]
            
            if "overlay_min_scale" in float_values:
                self.overlay_min_max_scale[0] = float_values["overlay_min_scale"]
            if "overlay_max_scale" in float_values:
                self.overlay_min_max_scale[1] = float_values["overlay_max_scale"]
            
            # Set details texts if available
            if "details_texts" in config_data:
                self.reorderable_sliders.set_details_values(config_data["details_texts"])
            
            # Set skip existing checkbox
            self.skip_existing_checkbox.setChecked(config_data.get("skip_existing", True))
            
            # Apply saved order if available
            if "augmentation_order" in config_data:
                self.reorderable_sliders.reorder_sliders_from_config(config_data["augmentation_order"])
            
            self.update_sliders_state()

    def scan_folders(self):
        dataset_root = self.parent.dataset_root
        
        # Clear previous data structures
        skip_categories = self.config_loader.get_skip_categories()
        self.skip_augmentations = {category: [] for category in skip_categories}
        
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
            
        # Get the number of skip categories + folder column + skip all column
        total_columns = len(skip_categories) + 2
        
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make folder names read-only
            self.skip_table.setItem(row, 0, folder_item)
            
            if hasattr(self.parent, 'image_viewer_tab'):
                list_item = QListWidgetItem(folder)
                self.parent.image_viewer_tab.folder_list.addItem(list_item)
                
            # Create checkboxes for each skip category + Skip All
            for col in range(1, total_columns):
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 0px; margin-right: auto;")  # Align checkbox to the left
                
                # Special handling for overlay if it exists
                category_idx = col - 1
                if category_idx < len(skip_categories) and skip_categories[category_idx] == "Overlay":
                    checkbox.setEnabled(False if not self.parent.overlay_image_dir else True)
                    
                # Connect Skip All checkbox
                if col == total_columns - 1:  # Last column is "Skip All"
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