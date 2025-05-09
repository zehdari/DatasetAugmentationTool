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

        # Store UI element references
        self.sliders = {}  # Slider references by slider_attr
        self.slider_values = {}  # Value references by value_attr
        self.child_sliders = {}  # Child slider references by attr
        self.float_inputs = {}  # Float input references by attr
        
        # Store parameter values dynamically
        self.param_values = {}  # All parameter values indexed by parameter name
        
        self.class_colors = {}
        self.id_to_label = {}
        
        # Initialize class colors manager
        self.class_colors_manager = ClassColorsManager(self)
        
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
            slider_attr = aug_info["slider_attr"]
            value_attr = aug_info["value_attr"]
            default_value = aug_info.get("default_value", 50)
            
            # Store default value
            self.param_values[f"{aug_type}_weights"] = [default_value, 100 - default_value]
            
            slider, value_edit, details_widget = self.reorderable_sliders.add_slider(
                name=aug_info["name"],
                slider_attr=slider_attr,
                value_attr=value_attr,
                default_value=default_value,
                augmentation_type=aug_type,
                details_text=aug_info.get("details", "")
            )
            
            # Store references for later use
            self.sliders[slider_attr] = slider
            self.slider_values[value_attr] = value_edit
        
        # Process child sliders based on configuration
        for parent_attr, children in child_sliders_config.items():
            for child_info in children:
                child_attr = child_info["attr"]
                child_name = child_info["name"]
                child_default = child_info.get("default", 50)
                
                # Store default value
                param_name = f"{child_attr.replace('_slider', '')}_weights"
                self.param_values[param_name] = [child_default, 100 - child_default]
                
                child_slider = self.reorderable_sliders.add_child_slider(
                    parent_attr=parent_attr,
                    name=child_name,
                    child_attr=child_attr,
                    default_value=child_default
                )
                
                # Store reference
                if child_slider:
                    self.child_sliders[child_attr] = child_slider
        
        # Process float inputs based on configuration
        for parent_attr, inputs in float_params_config.items():
            for input_info in inputs:
                input_attr = input_info["attr"]
                input_name = input_info["name"]
                default_value = input_info.get("default", 0.5)
                min_value = input_info.get("min", 0.0)
                max_value = input_info.get("max", 1.0)
                step = input_info.get("step", 0.01)
                
                # Store default value directly in param_values
                self.param_values[input_attr] = default_value
                
                float_input = self.reorderable_sliders.add_float_input(
                    parent_attr=parent_attr,
                    name=input_name,
                    input_attr=input_attr,
                    default_value=default_value,
                    min_value=min_value,
                    max_value=max_value,
                    step=step
                )
                
                # Store reference
                if float_input:
                    self.float_inputs[input_attr] = float_input
        
        # Connect signals
        self.reorderable_sliders.childSliderValueChanged.connect(self.on_child_slider_value_changed)
        self.reorderable_sliders.floatValueChanged.connect(self.on_float_value_changed)
        self.reorderable_sliders.sliderValueChanged.connect(self.on_slider_value_changed)
        
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

    def on_slider_value_changed(self, slider_attr, value):
        """Handle parent slider value changes."""
        # Find augmentation type for this slider
        aug_type = self.find_aug_type_for_slider(slider_attr)
        if aug_type:
            # Update weights parameter
            self.param_values[f"{aug_type}_weights"] = [value, 100 - value]

    def find_aug_type_for_slider(self, slider_attr):
        """Find the augmentation type associated with a slider attribute."""
        augmentations = self.config_loader.get_augmentations()
        for aug_type, aug_info in augmentations.items():
            if aug_info["slider_attr"] == slider_attr:
                return aug_type
        return None

    def on_child_slider_value_changed(self, child_attr, value):
        """Handle child slider value changes."""
        # Update parameter value based on the child slider attribute
        param_name = f"{child_attr.replace('_slider', '')}_weights"
        self.param_values[param_name] = [value, 100 - value]
            
    def on_float_value_changed(self, input_attr, value):
        """Handle float input value changes."""
        # Simply store the value directly by its attribute name
        self.param_values[input_attr] = value

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
        
        # Initialize parameters dictionary with common parameters
        params = {
            'skip_existing': self.skip_existing_checkbox.isChecked(),
            'skip_augmentations': self.get_skip_augmentations(),
            'augmentation_order': augmentation_order
        }
        
        # Add all parameter values to the params dictionary
        for param_name, value in self.param_values.items():
            params[param_name] = value
        
        # Only special case: handle overlay when directory is not selected
        if "overlay_weights" in params and not self.parent.overlay_image_dir:
            params["overlay_weights"] = [0, 100]  # Disable overlay
            
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
        """Save the current configuration to a YAML file"""
        # Get the current order of sliders
        augmentation_order = self.get_augmentation_order()
        
        # Get all slider values
        slider_values = self.reorderable_sliders.get_slider_values()
        
        # Get all float input values
        float_values = self.reorderable_sliders.get_float_values()
        
        # Get skip augmentations
        skip_augmentations = self.get_skip_augmentations()
        
        # Create config data dictionary
        config_data = {
            "augmentation_order": augmentation_order,
            "skip_existing": self.skip_existing_checkbox.isChecked(),
            "skip_augmentations": skip_augmentations
        }
        
        # Add all slider values
        slider_values_section = {}
        for slider_attr, value in slider_values.items():
            # Use more readable names in the config
            config_key = slider_attr.replace("_slider", "_probability")
            slider_values_section[config_key] = value
        
        config_data["slider_values"] = slider_values_section
            
        # Add all float values
        float_values_section = {}
        for float_attr, value in float_values.items():
            float_values_section[float_attr] = value
        
        config_data["float_values"] = float_values_section
            
        # Save to YAML file
        self.parent.config_manager.save_config(config_data)

    def load_existing_config(self):
        """Load configuration from a YAML file"""
        config_data = self.parent.config_manager.load_config()
        if config_data:
            # Get the known augmentations
            augmentations = self.config_loader.get_augmentations()
            
            # Set slider values for known augmentations
            slider_values = {}
            
            # Handle direct slider values or those in a nested section
            if "slider_values" in config_data:
                # New format with nested sections
                slider_values_data = config_data["slider_values"]
            else:
                # Old format with flat structure
                slider_values_data = config_data
            
            # Process augmentation sliders
            for aug_type, aug_info in augmentations.items():
                slider_attr = aug_info["slider_attr"]
                config_key = slider_attr.replace("_slider", "_probability")
                
                if config_key in slider_values_data:
                    slider_values[slider_attr] = slider_values_data[config_key]
                    
            # Process child slider values
            child_sliders_config = self.config_loader.get_child_sliders()
            for parent_attr, children in child_sliders_config.items():
                for child_info in children:
                    child_attr = child_info["attr"]
                    config_key = child_attr.replace("_slider", "_probability")
                    
                    if config_key in slider_values_data:
                        slider_values[child_attr] = slider_values_data[config_key]
            
            # Update all slider values
            self.reorderable_sliders.set_slider_values(slider_values)
            
            # Set float input values
            float_values = {}
            
            # Handle direct float values or those in a nested section
            if "float_values" in config_data:
                # New format with nested sections
                float_values_data = config_data["float_values"]
            else:
                # Old format with flat structure
                float_values_data = config_data
                
            float_params_config = self.config_loader.get_float_params()
            for parent_attr, inputs in float_params_config.items():
                for input_info in inputs:
                    input_attr = input_info["attr"]
                    if input_attr in float_values_data:
                        float_values[input_attr] = float_values_data[input_attr]
            
            # Update all float input values
            self.reorderable_sliders.set_float_values(float_values)
            
            # Update parameter values in memory
            self.update_param_values_from_sliders()
            
            # Set skip existing checkbox
            self.skip_existing_checkbox.setChecked(config_data.get("skip_existing", True))
            
            # Set skip augmentations if available
            if "skip_augmentations" in config_data:
                self.skip_augmentations = config_data["skip_augmentations"]
                self.update_skip_table()
            
            # Apply saved order if available
            if "augmentation_order" in config_data:
                self.reorderable_sliders.reorder_sliders_from_config(config_data["augmentation_order"])
            
            self.update_sliders_state()
            
    def update_skip_table(self):
        """Update the skip table based on current skip_augmentations settings"""
        skip_categories = self.config_loader.get_skip_categories()
        
        # Map skip category names to column indices
        category_to_col = {category: idx + 1 for idx, category in enumerate(skip_categories)}
        
        # Update checkboxes in the skip table
        for row in range(self.skip_table.rowCount()):
            folder_name = self.skip_table.item(row, 0).text()
            
            # Check if this folder should be skipped for all categories
            skip_all = True
            for category in skip_categories:
                if folder_name not in self.skip_augmentations.get(category, []):
                    skip_all = False
                    break
            
            # Set the "Skip All" checkbox
            skip_all_col = len(skip_categories) + 1
            skip_all_checkbox = self.skip_table.cellWidget(row, skip_all_col)
            if skip_all_checkbox:
                skip_all_checkbox.setChecked(skip_all)
                
            # Set individual category checkboxes
            for category, col in category_to_col.items():
                checkbox = self.skip_table.cellWidget(row, col)
                if checkbox:
                    is_skipped = folder_name in self.skip_augmentations.get(category, [])
                    checkbox.setChecked(is_skipped)
                    checkbox.setEnabled(not skip_all)
            
    def update_param_values_from_sliders(self):
        """Update internal parameter values from slider states."""
        # Get current slider values
        slider_values = self.reorderable_sliders.get_slider_values()
        float_values = self.reorderable_sliders.get_float_values()
        
        # Update parent slider parameters
        augmentations = self.config_loader.get_augmentations()
        for aug_type, aug_info in augmentations.items():
            slider_attr = aug_info["slider_attr"]
            if slider_attr in slider_values:
                value = slider_values[slider_attr]
                self.param_values[f"{aug_type}_weights"] = [value, 100 - value]
        
        # Update child slider parameters
        child_sliders_config = self.config_loader.get_child_sliders()
        for parent_attr, children in child_sliders_config.items():
            for child_info in children:
                child_attr = child_info["attr"]
                if child_attr in slider_values:
                    value = slider_values[child_attr]
                    param_name = f"{child_attr.replace('_slider', '')}_weights"
                    self.param_values[param_name] = [value, 100 - value]
        
        # Update float input parameters - directly store by attribute name
        for input_attr, value in float_values.items():
            self.param_values[input_attr] = value

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