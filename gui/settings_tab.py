from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFileDialog, QSlider, QMessageBox, QGroupBox, QFormLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, 
                             QSplitter, QListWidget, QSizePolicy, QListWidgetItem, 
                             QColorDialog, QAbstractItemView, QProgressDialog, 
                             QProgressBar, QTextEdit, QLineEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import random
import time
import os
import re
from utils.ui_components import ClickFilter, CustomLineEdit
from utils.dataset_manager import DatasetManager
import yaml

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
        self.label_to_id = {}
        
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
        for slider_info in self.slider_data:
            self.add_slider_to_list(slider_info["name"], slider_info["object"], slider_info["value_object"])
        
        weights_layout.addWidget(self.sliders_list)
        
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

        self.class_color_group = QGroupBox("Class Colors")
        self.class_colors_layout = QVBoxLayout()
        self.class_colors_table = QTableWidget()
        self.class_colors_table.setColumnCount(2)
        self.class_colors_table.setHorizontalHeaderLabels(['Class', 'Color'])
        self.class_colors_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_colors_table.itemClicked.connect(self.on_color_cell_clicked)
        self.class_colors_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.class_colors_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.class_colors_layout.addWidget(self.class_colors_table)
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

    def add_slider_to_list(self, name, slider_attr, value_attr):
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
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setEnabled(False)
        value_label = CustomLineEdit("50")
        value_label.setFixedWidth(40)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setEnabled(False)
        slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
        value_label.textChanged.connect(lambda text, sld=slider: sld.setValue(int(text)) if text.isdigit() else None)
        return slider, value_label

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

        for row in range(self.skip_table.rowCount()):
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            if overlay_checkbox:
                overlay_checkbox.setEnabled(enable_overlay_sliders)

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
            'mirror_weights': [self.mirror_slider.value(), 100 - self.mirror_slider.value()],
            'crop_weights': [self.crop_slider.value(), 100 - self.crop_slider.value()],
            'zoom_weights': [self.zoom_slider.value(), 100 - self.zoom_slider.value()],
            'rotate_weights': [self.rotate_slider.value(), 100 - self.rotate_slider.value()],
            'overlay_weights': ([self.overlay_slider.value(), 100 - self.overlay_slider.value()] 
                               if self.parent.overlay_image_dir else [0, 100]),
            'rotation_random_vs_90_weights': [self.rotation_random_vs_90_slider.value(), 
                                             100 - self.rotation_random_vs_90_slider.value()],
            'overlay_min_max_scale': self.overlay_min_max_scale,
            'maintain_aspect_ratio_weights': [self.maintain_aspect_ratio_slider.value(),
                                            100 - self.maintain_aspect_ratio_slider.value()],
            'zoom_in_vs_out_weights': [self.zoom_in_vs_out_slider.value(),
                                      100 - self.zoom_in_vs_out_slider.value()],
            'zoom_padding': self.zoom_padding,
            'augmentation_order': augmentation_order
        }
        
        return params

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
                    color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
                    if color.isValid():
                        self.class_colors[class_id] = color
                        self.update_class_colors_table()
                        # Update the image in the viewer tab if available
                        if hasattr(self.parent, 'image_viewer_tab'):
                            self.parent.image_viewer_tab.show_image()
        except Exception as e:
            print(f"Error in on_color_cell_clicked: {str(e)}")
            pass

    def update_class_colors_table(self):
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

    def scan_folders(self):
        dataset_root = self.parent.dataset_root
        
        # Clear previous data structures
        for key in self.skip_augmentations.keys():
            self.skip_augmentations[key] = []
        
        # Clear class mappings
        self.id_to_label = {}
        self.label_to_id = {}
        self.class_colors = {}

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

    def toggle_skip_all(self, state, row):
        skip_all_checked = state == Qt.CheckState.Checked
        for col in range(1, 6):  # Update to check relevant columns
            checkbox = self.skip_table.cellWidget(row, col)
            checkbox.setEnabled(not skip_all_checked)
        if not self.parent.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            overlay_checkbox.setEnabled(False)

    def setup_progress_dialog(self, progress_dialog):
        """Set up the progress dialog for augmentation process"""
        layout = QVBoxLayout(progress_dialog)
        
        # Top header with progress and time
        header_layout = QHBoxLayout()
        
        # Progress label on the left
        self.progress_label = QLabel("Starting...")
        header_layout.addWidget(self.progress_label, 1)  # Give it stretch factor
        
        # Elapsed time in the top right with label
        elapsed_layout = QHBoxLayout()
        elapsed_time_descriptor = QLabel("Elapsed Time:")
        self.elapsed_time_label = QLabel("0:00")
        self.elapsed_time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        elapsed_layout.addWidget(elapsed_time_descriptor)
        elapsed_layout.addWidget(self.elapsed_time_label)
        header_layout.addLayout(elapsed_layout)
        
        layout.addLayout(header_layout)
        
        # Time remaining estimate
        self.time_label = QLabel("Estimated remaining: Calculating...")
        layout.addWidget(self.time_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Add text area for logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        layout.addWidget(self.log_text)
        
        # Add our single cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.handle_cancellation)
        layout.addWidget(cancel_button)

        # Remove the default label from QProgressDialog
        progress_dialog.findChild(QLabel).hide()

    def update_progress(self, value):
        """Update progress and time estimates using linear extrapolation."""
        # Skip if cancelled or no progress yet
        if self.is_cancelled or value <= 0:
            return

        try:
            current_time = time.time()
            elapsed = current_time - self.start_time

            # Update elapsed display
            self.elapsed_time_label.setText(self.format_elapsed_time(elapsed))

            # Update progress %
            self.progress_label.setText(f"Progress: {value}%")

            # Throttle ETA updates to once a second
            if current_time - self.last_time_update >= 1.0:
                done = self.parent.worker.processed_files
                total = self.parent.worker.total_files

                if done > 0 and total > done:
                    avg_per_file = elapsed / done
                    remaining = (total - done) * avg_per_file
                else:
                    remaining = 0.0

                self.time_label.setText(
                    f"Estimated remaining: {self.format_elapsed_time(remaining)}"
                )
                self.last_time_update = current_time

        except RuntimeError:
            # Widget deleted, ignore
            pass

    def handle_cancellation(self):
        """Handle user cancellation of the augmentation process"""
        self.is_cancelled = True
        self.parent.worker.cancel()
        self.log_text.append("\nCancelling...")
        self.parent.progress_dialog.close()

    def format_elapsed_time(self, seconds):
        """Format elapsed time into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:d}:{secs:02d}"

    def save_current_config(self):
        # Get the current order of sliders
        slider_order = []
        for i in range(self.sliders_list.count()):
            item_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
            # Find the slider widget in the layout
            for j in range(item_widget.layout().count()):
                widget = item_widget.layout().itemAt(j).widget()
                if isinstance(widget, QSlider):
                    # Find which attribute this slider corresponds to
                    for attr_name, attr_value in vars(self).items():
                        if attr_value is widget and attr_name in self.slider_to_augmentation_type:
                            slider_order.append(self.slider_to_augmentation_type[attr_name])
                            break
                    break
        
        config_data = {
            "augmentation_order": slider_order,
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
            self.crop_slider.setValue(config_data.get("crop_probability", 0))
            self.maintain_aspect_ratio_slider.setValue(config_data.get("maintain_aspect_ratio", 0))
            self.mirror_slider.setValue(config_data.get("mirror_probability", 0))
            self.overlay_slider.setValue(config_data.get("overlay_probability", 0))
            self.rotate_slider.setValue(config_data.get("rotate_probability", 0))
            self.rotation_random_vs_90_slider.setValue(config_data.get("rotation_random_vs_90", 0))
            self.zoom_in_vs_out_slider.setValue(config_data.get("zoom_in_vs_out", 0))
            self.zoom_slider.setValue(config_data.get("zoom_probability", 0))
            self.skip_existing_checkbox.setChecked(config_data.get("skip_existing", False))
            
            # Apply saved order if available
            if "augmentation_order" in config_data:
                self.reorder_sliders_from_config(config_data["augmentation_order"])
            
            self.update_sliders_state()
                
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

    def atoi(self, text):
        """Helper function for natural sort order"""
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        """Sort text strings in natural order"""
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]