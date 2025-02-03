import sys
import os
import re
import random
import time
from collections import Counter
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QSlider, QSpinBox, QMessageBox, QGroupBox, QFormLayout,
                             QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QGridLayout, QSplitter,
                             QListWidget, QListWidgetItem, QSizePolicy, QScrollArea, QTabWidget, QColorDialog, 
                             QAbstractItemView, QProgressDialog, QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QObject, QEvent, QSize, QPointF, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QImage
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from augment_data import augment_image
import multiprocessing
import concurrent.futures
import gc
from queue import Queue
from dataclasses import dataclass
from typing import List, Dict, Any
import hashlib
import json
import yaml

@dataclass
class BatchItem:
    root: str
    file: str
    current_subfolder: str
    current_augmented_image_dir: str
    current_augmented_label_dir: str
    params: Dict[str, Any]

def process_batch(batch: List[BatchItem]) -> List[tuple[bool, str]]:
    """Process a batch of images in parallel"""
    results = []
    for item in batch:
        try:
            # Setup paths
            img_path = os.path.join(item.root, item.file)
            lbl_path = os.path.join(item.params['label_dir'],
                                os.path.relpath(img_path, item.params['image_dir'])
                                ).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                                
            # Check if we should skip this image
            augmented_img_path = os.path.join(item.current_augmented_image_dir, item.file)
            augmented_lbl_path = os.path.join(item.current_augmented_label_dir,
                                            item.file.replace('.jpg', '.txt')
                                            .replace('.jpeg', '.txt')
                                            .replace('.png', '.txt'))
                                            
            if item.params.get('skip_existing', True) and os.path.exists(augmented_img_path) and os.path.exists(augmented_lbl_path):
                results.append((True, f"Skipped existing: {item.file}"))
                continue

            # Read and process image
            image = cv2.imread(img_path)
            if image is None:
                results.append((False, f"Could not read image: {img_path}"))
                continue
                
            (h, w) = image.shape[:2]

            # Process labels
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    polygons = []
                    class_ids = []
                    for line in lines:
                        parts = line.strip().split()
                        class_id = parts[0]
                        class_ids.append(class_id)
                        polygon = [(float(parts[i]), float(parts[i + 1])) 
                                for i in range(1, len(parts), 2)]
                        polygons.append(polygon)
            else:
                results.append((False, f"Label file not found: {lbl_path}"))
                continue

            # Extract augmentation parameters
            augmentation_params = {
                'skip_augmentations': item.params['skip_augmentations'],
                'mirror_weights': item.params['mirror_weights'],
                'crop_weights': item.params['crop_weights'],
                'overlay_weights': item.params['overlay_weights'],
                'rotate_weights': item.params['rotate_weights'],
                'rotation_random_vs_90_weights': item.params['rotation_random_vs_90_weights'],
                'overlay_min_max_scale': item.params['overlay_min_max_scale'],
                'maintain_aspect_ratio_weights': item.params['maintain_aspect_ratio_weights'],
                'zoom_weights': item.params['zoom_weights'],
                'zoom_in_vs_out_weights': item.params['zoom_in_vs_out_weights'],
                'zoom_padding': item.params['zoom_padding'],
                'coco_image_folder': item.params['coco_image_folder']
            }

            # Perform augmentation
            from augment_data import augment_image
            augmented_image, augmented_polygons = augment_image(
                image=image,
                polygons=polygons,
                current_subfolder=item.current_subfolder,
                class_ids=class_ids,
                h=h,
                w=w,
                **augmentation_params
            )

            # Save results
            os.makedirs(os.path.dirname(augmented_img_path), exist_ok=True)
            cv2.imwrite(augmented_img_path, augmented_image)
            
            os.makedirs(os.path.dirname(augmented_lbl_path), exist_ok=True)
            with open(augmented_lbl_path, 'w') as f:
                for polygon in augmented_polygons:
                    line = ' '.join(map(str, [polygon[0]] + 
                                    [coord for point in polygon[1:] for coord in point]))
                    f.write(line + '\n')
            
            results.append((True, f"Successfully processed: {item.file}"))
            
        except Exception as e:
            results.append((False, f"Error processing {item.file}: {str(e)}"))
            
    return results

def process_single_image_worker(args):
    """Standalone function for processing a single image."""
    try:
        relative_path, label_path, subfolder, params = args
        
        # Set consistent seed for this file
        seed = int(hashlib.md5(relative_path.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        
        # Construct full paths
        img_path = os.path.join(params['image_dir'], relative_path)
        
        # Construct output paths maintaining folder structure
        augmented_img_path = os.path.join(params['augmented_image_dir'], relative_path)
        augmented_lbl_path = os.path.join(
            params['augmented_label_dir'],
            os.path.splitext(relative_path)[0] + '.txt'
        )
        
        # Create output directories
        os.makedirs(os.path.dirname(augmented_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(augmented_lbl_path), exist_ok=True)
        
        # Check if we should skip this image
        if (params.get('skip_existing', True) and 
            os.path.exists(augmented_img_path) and 
            os.path.exists(augmented_lbl_path)):
            return True, f"Skipped existing: {relative_path}"
        
        # Read and validate image
        image = cv2.imread(img_path)
        if image is None:
            return False, f"Could not read image: {img_path}"
        
        h, w = image.shape[:2]
        
        # Read and parse labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            polygons = []
            class_ids = []
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]
                class_ids.append(class_id)
                coords = [(float(parts[i]), float(parts[i + 1])) 
                         for i in range(1, len(parts), 2)]
                polygons.append(coords)
        
        # Perform augmentation
        augmented_image, augmented_polygons = augment_image(
            image=image,
            polygons=polygons,
            current_subfolder=subfolder,
            class_ids=class_ids,
            h=h,
            w=w,
            **{k: params[k] for k in [
                'skip_augmentations', 'mirror_weights', 'crop_weights',
                'overlay_weights', 'rotate_weights',
                'rotation_random_vs_90_weights',
                'overlay_min_max_scale', 'maintain_aspect_ratio_weights',
                'zoom_weights', 'zoom_in_vs_out_weights', 'zoom_padding',
                'coco_image_folder'
            ]}
        )
        
        # Save augmented image
        cv2.imwrite(augmented_img_path, augmented_image)
        
        # Save augmented labels
        with open(augmented_lbl_path, 'w') as f:
            for polygon in augmented_polygons:
                line = ' '.join(map(str, [polygon[0]] + 
                               [coord for point in polygon[1:] for coord in point]))
                f.write(line + '\n')
        
        return True, f"Successfully processed: {relative_path}"
        
    except Exception as e:
        return False, f"Error processing {relative_path}: {str(e)}"

class AugmentationWorker(QThread):
    progress = pyqtSignal(int)
    progress_log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_cancelled = False
        self.total_files = 0
        self.processed_files = 0
        self.semaphore = None
        
    def calculate_optimal_workers(self):
        """Calculate the optimal number of worker processes based on system resources."""
        try:
            import psutil
            
            # Get system memory information
            mem = psutil.virtual_memory()
            available_memory = mem.available
            
            # Sample the first image to estimate memory usage
            image_label_pairs = self.collect_image_label_pairs()
            if not image_label_pairs:
                return os.cpu_count() or 1
                
            img_path = os.path.join(self.params['image_dir'], image_label_pairs[0][0])
            img_size = os.path.getsize(img_path)
            label_size = os.path.getsize(image_label_pairs[0][1])
            
            # Estimate memory needed per process (4x for safety margin)
            memory_per_process = (img_size + label_size) * 4
            
            # Calculate max workers based on available memory (use 70% of available memory)
            memory_based_workers = int((available_memory * 0.7) // memory_per_process)
            
            # Get CPU count
            cpu_count = os.cpu_count() or 1
            
            # Use the minimum of memory-based or CPU-based worker count
            return max(1, min(memory_based_workers, cpu_count))
            
        except Exception as e:
            self.progress_log.emit(f"Error calculating workers: {str(e)}")
            return os.cpu_count() or 1
    
    def run(self):
        """Execute the augmentation process with optimized parallel processing."""
        try:
            # Collect image-label pairs
            image_label_pairs = self.collect_image_label_pairs()
            
            if not image_label_pairs:
                self.progress_log.emit("No image files found to process.")
                self.finished.emit()
                return
            
            self.total_files = len(image_label_pairs)
            self.progress_log.emit(f"Total files to process: {self.total_files}")
            
            # Calculate optimal number of workers
            num_workers = self.calculate_optimal_workers()
            self.progress_log.emit(f"Using {num_workers} worker processes")
            
            # Initialize processing
            self.processed_files = 0
            
            # Process images in parallel with proper resource management
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                
                # Submit all tasks
                for relative_path, label_path, subfolder in image_label_pairs:
                    if self.is_cancelled:
                        break
                    
                    future = executor.submit(
                        process_single_image_worker,
                        (relative_path, label_path, subfolder, self.params)
                    )
                    futures.append(future)
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    if self.is_cancelled:
                        break
                    
                    try:
                        success, message = future.result()
                        self.processed_files += 1
                        progress = int(self.processed_files * 100 / self.total_files)
                        self.progress.emit(progress)
                        
                        if success:
                            self.progress_log.emit(f"[{self.processed_files}/{self.total_files}] {message}")
                        else:
                            self.progress_log.emit(f"Error: {message}")
                            
                    except Exception as e:
                        self.error.emit(str(e))
                        continue
            
            if not self.is_cancelled:
                self.progress_log.emit("\nProcessing complete!")
                self.finished.emit()
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Clean up resources
            gc.collect()

    def atoi(self, text):
        """Helper function for natural sort order"""
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        """Sort text strings in natural order"""
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def collect_image_label_pairs(self):
        """Collect matching image and label file pairs with proper relative paths."""
        image_label_pairs = []
        for root, _, files in os.walk(self.params['image_dir']):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Get relative path from image directory
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, self.params['image_dir'])
                    subfolder = os.path.dirname(relative_path)
                    
                    # Construct label path maintaining folder structure
                    label_path = os.path.join(
                        self.params['label_dir'],
                        os.path.splitext(relative_path)[0] + '.txt'
                    )
                    
                    if os.path.exists(label_path):
                        image_label_pairs.append((relative_path, label_path, subfolder))
        
        return sorted(image_label_pairs, key=lambda x: self.natural_keys(x[0]))

    def cancel(self):
        """Cancel the augmentation process."""
        self.is_cancelled = True
        self.progress_log.emit("\nCancelling...")
    
class ImageCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get_image(self, path):
        if path in self.cache:
            return self.cache[path].copy()
        
        image = cv2.imread(path)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = image
        return image.copy()

    def clear(self):
        self.cache.clear()

class ClickFilter(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            for widget in obj.findChildren(QLineEdit):
                widget.clearFocus()
        return super().eventFilter(obj, event)

class CustomLineEdit(QLineEdit):
    def focusOutEvent(self, event):
        self.deselect()
        super().focusOutEvent(event)

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file

    def save_config(self, config_data):
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Configuration", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    json.dump(config_data, file, indent=4)
                QMessageBox.information(None, "Success", f"Configuration saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to save configuration: {e}")

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Load Configuration", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    config_data = json.load(file)
                QMessageBox.information(None, "Success", f"Configuration loaded from {file_path}")
                return config_data
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load configuration: {e}")
        return None

def parse_dataset_yaml(dataset_root):
    """
    Search for and parse YOLO dataset YAML file in the dataset root
    """
    # Find all .yaml files in the dataset root
    yaml_files = [f for f in os.listdir(dataset_root) if f.lower().endswith('.yaml')]
    
    for filename in yaml_files:
        yaml_path = os.path.join(dataset_root, filename)
        try:
            with open(yaml_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
                
                # Multiple ways to extract class names
                if 'names' in yaml_data:
                    # Direct names list
                    if isinstance(yaml_data['names'], list):
                        return yaml_data['names']
                    # Names mapped from integers
                    elif isinstance(yaml_data['names'], dict):
                        return list(yaml_data['names'].values())
                
                # Alternative extraction methods
                if isinstance(yaml_data, dict):
                    keys_to_check = ['nc_names', 'class_names', 'classes']
                    for key in keys_to_check:
                        if key in yaml_data and isinstance(yaml_data[key], list):
                            return yaml_data[key]
        except Exception as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")

class AugmentDatasetGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.image_cache = ImageCache(max_size=100)
        self.config_manager = ConfigManager('augmentation_config.json')

        self.dataset_root = ""
        self.overlay_image_dir = ""
        self.output_dir = ""
        self.skip_augmentations = {
            'Zoom': [],
            'Crop': [],
            'Rotate': [],
            'Mirror': [],
            'Overlay': []
        }
        self.image_paths = []  # List to store image paths
        self.folder_images = []
        self.label_paths = {}  # Dictionary to store label paths
        self.current_image_index = 0  # To store the index of the current displayed image
        self.current_image_path = ""
        self.folder_name = ""
        self.class_colors = {}  # Dictionary to store class colors
        self.augmented_image = None
        self.augmented_polygons = None
        self.augmented_image_original_dims = None

        self.rotation_random_vs_90 = [25, 75]
        self.zoom_in_vs_out_weights = [40, 60]
        self.zoom_in_min_padding = 0.05
        self.zoom_in_max_padding = 0.5
        self.zoom_out_min_padding = 0.1
        self.zoom_out_max_padding = 0.8
        self.zoom_padding = [self.zoom_in_min_padding, self.zoom_in_max_padding, self.zoom_out_min_padding, self.zoom_out_max_padding]
        self.maintain_aspect_ratio_weights = [50, 50]
        self.overlay_min_max_scale = [0.3, 1.0]

        self.show_labels = True
        self.show_polygons = True 
        self.show_bounding_boxes = False 
        self.show_points = False 

        self.output_dir_set = False  # Flag to track if output directory has been set

        self.show_original = False
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dataset Augmentation GUI')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        # Tabs
        tab_widget = QTabWidget()
        self.augmentation_settings_tab = QWidget()
        self.image_viewer_tab = QWidget()
        self.dataset_stats_tab = QWidget()

        tab_widget.addTab(self.augmentation_settings_tab, "Settings")
        tab_widget.addTab(self.image_viewer_tab, "Image Viewer")
        tab_widget.addTab(self.dataset_stats_tab, "Dataset Stats")

        self.init_augmentation_settings_tab()
        self.init_image_viewer_tab()
        self.init_dataset_stats_tab()

        main_layout.addWidget(tab_widget)

        # Bottom button layout
        bottom_button_layout = QHBoxLayout()  # Use vertical layout to stack elements vertically

        # Horizontal layout for Save and Load buttons

        # Run button on its own row
        self.run_btn = QPushButton("Run Augmentation")
        self.run_btn.clicked.connect(self.run_augmentation)

        bottom_button_layout.addWidget(self.run_btn)          # Second row with Run button

        # Add bottom button layout to main layout
        main_layout.addLayout(bottom_button_layout)


        self.setLayout(main_layout)
        self.installEventFilter(ClickFilter(self))

    def save_current_config(self):
        config_data = {
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
        self.config_manager.save_config(config_data)

    def load_existing_config(self):
        config_data = self.config_manager.load_config()
        if config_data:
            self.crop_slider.setValue(config_data.get("crop_probability", 0))
            self.maintain_aspect_ratio_slider.setValue(config_data.get("maintain_aspect_ratio", 0))
            self.mirror_slider.setValue(config_data.get("mirror_probability", 0))
            self.overlay_slider.setValue(config_data.get("overlay_probability", 0))
            self.rotate_slider.setValue(config_data.get("rotate_probability", 0))
            self.rotation_random_vs_90_slider.setValue(config_data.get("rotation_random_vs_90", 0))
            self.zoom_in_vs_out_slider.setValue(config_data.get("zoom_in_vs_out", 0))
            self.zoom_slider.setValue(config_data.get("zoom_probability", 0))
            self.skip_existing_checkbox.setChecked(config_data.get("skip_existing", False))


    def init_augmentation_settings_tab(self):
        layout = QVBoxLayout()

        # Combined directory selection
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
        
        # Weights sliders with scroll area inside a group box
        weights_group = QGroupBox("Augmentation Settings")
        weights_layout = QGridLayout()

        self.mirror_slider, self.mirror_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Mirror % Probability:", self.mirror_slider, self.mirror_value, 0)

        self.rotate_slider, self.rotate_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Rotate % Probability:", self.rotate_slider, self.rotate_value, 1)

        self.rotation_random_vs_90_slider, self.rotation_random_vs_90_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Rotation (0 to 360) vs 90 %: ", self.rotation_random_vs_90_slider, self.rotation_random_vs_90_value, 2)

        self.crop_slider, self.crop_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Crop % Probability:", self.crop_slider, self.crop_value, 3)

        self.maintain_aspect_ratio_slider, self.maintain_aspect_ratio_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Maintain Aspect Ratio on Crop %: ", self.maintain_aspect_ratio_slider, self.maintain_aspect_ratio_value, 4)

        self.zoom_slider, self.zoom_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Zoom % Probability:", self.zoom_slider, self.zoom_value, 5)

        self.zoom_in_vs_out_slider, self.zoom_in_vs_out_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Zoom In vs Out %: ", self.zoom_in_vs_out_slider, self.zoom_in_vs_out_value, 6)

        self.overlay_slider, self.overlay_value = self.create_slider()
        self.add_slider_to_layout(weights_layout, "Overlay % Probability:", self.overlay_slider, self.overlay_value, 7)

        settings_buttons_group = QHBoxLayout()

        # Skip existing checkbox
        self.skip_existing_checkbox = QCheckBox("Skip Already Augmented Images")
        self.skip_existing_checkbox.setChecked(True)
        settings_buttons_group.addWidget(self.skip_existing_checkbox)

        
        self.load_config_button = QPushButton("Load Config")
        self.save_config_button = QPushButton("Save Config")
        
        self.load_config_button.clicked.connect(self.load_existing_config)
        self.save_config_button.clicked.connect(self.save_current_config)

       
        settings_buttons_group.addWidget(self.load_config_button)
        settings_buttons_group.addWidget(self.save_config_button)

        weights_layout.addLayout(settings_buttons_group, weights_layout.rowCount(), 0, 1, weights_layout.columnCount())
        weights_group.setLayout(weights_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(weights_layout)
        scroll_area.setWidget(scroll_widget)

        group_box_with_scroll = QGroupBox("Augmentation Settings")
        group_box_layout = QVBoxLayout()
        group_box_layout.addWidget(scroll_area)
        group_box_with_scroll.setLayout(group_box_layout)
        group_box_with_scroll.setMinimumWidth(400)

        weights_skip_layout.addWidget(group_box_with_scroll)
        skip_colors_layout = QSplitter(Qt.Orientation.Vertical)

        # Skip Augmentations
        self.skip_group = QGroupBox("Skip Augmentations for Folders")
        self.skip_layout = QVBoxLayout()
        self.skip_table = QTableWidget()
        self.skip_table.setColumnCount(7)  # Update the column count
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
        self.class_colors_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Set size policy to Expanding
        self.class_colors_layout.addWidget(self.class_colors_table)
        self.class_color_group.setLayout(self.class_colors_layout)
        
        skip_colors_layout.addWidget(self.class_color_group)
        skip_colors_layout.setCollapsible(0, False)
        skip_colors_layout.setCollapsible(1, False)

        weights_skip_layout.addWidget(skip_colors_layout)
        weights_skip_layout.setSizes([800, 400])  # Initial sizes of the panels, evenly split
        weights_skip_layout.setCollapsible(0, False)
        weights_skip_layout.setCollapsible(1, False)

        # Set minimum sizes
        self.skip_group.setMinimumWidth(400)  # Set a minimum width for the skip group
        weights_skip_layout.setMinimumWidth(1200)  # Ensure the splitter does not resize smaller than the initial setup
        weights_skip_layout.setHandleWidth(10)

        layout.addWidget(weights_skip_layout)
        self.augmentation_settings_tab.setLayout(layout)

    def init_image_viewer_tab(self):
        layout = QVBoxLayout()

        self.image_viewer_layout = QVBoxLayout()
        folder_list_layout = QHBoxLayout()
        folder_and_button_layout = QVBoxLayout()

        self.folder_list = QListWidget()
        self.folder_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.folder_list.setMaximumHeight(125)
        self.folder_list.setMinimumHeight(50)
        self.folder_list.itemClicked.connect(self.display_images)
        folder_and_button_layout.addWidget(self.folder_list)

        self.show_original_btn = QPushButton("Show Original Image")
        self.show_original_btn.clicked.connect(self.show_original_image)
        folder_and_button_layout.addWidget(self.show_original_btn)

        folder_list_layout.addLayout(folder_and_button_layout)

        checkboxes_button_layout = QVBoxLayout()
        self.labels_checkbox = QCheckBox("Show Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self.toggle_labels)
        checkboxes_button_layout.addWidget(self.labels_checkbox)

        self.polygons_checkbox = QCheckBox("Show Polygons")
        self.polygons_checkbox.setChecked(True)
        self.polygons_checkbox.stateChanged.connect(self.toggle_polygons)
        checkboxes_button_layout.addWidget(self.polygons_checkbox)

        self.bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(False)
        self.bbox_checkbox.stateChanged.connect(self.toggle_bounding_boxes)
        checkboxes_button_layout.addWidget(self.bbox_checkbox)

        self.points_checkbox = QCheckBox("Show Points")
        self.points_checkbox.setChecked(False)
        self.points_checkbox.stateChanged.connect(self.toggle_points)
        checkboxes_button_layout.addWidget(self.points_checkbox)

        self.augment_single_btn = QPushButton("Preview Augmentation")
        self.augment_single_btn.clicked.connect(self.augment_current_image)
        checkboxes_button_layout.addWidget(self.augment_single_btn)

        self.save_preview_btn = QPushButton("Save Current Preview")
        self.save_preview_btn.clicked.connect(self.save_current_preview)
        checkboxes_button_layout.addWidget(self.save_preview_btn)

        folder_list_layout.addLayout(checkboxes_button_layout)

        folder_list_layout.setStretch(0, 1)
        folder_list_layout.setStretch(1, 0)

        self.image_viewer_layout.addLayout(folder_list_layout)

        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_viewer_layout.addWidget(self.image_name_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumHeight(150)
        self.image_viewer_layout.addWidget(self.image_label)

        self.image_navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        self.image_slider = QSlider(Qt.Orientation.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.valueChanged.connect(self.slider_value_changed)

        self.image_navigation_layout.addWidget(self.prev_button)
        self.image_navigation_layout.addWidget(self.image_slider)
        self.image_navigation_layout.addWidget(self.next_button)

        self.image_viewer_layout.addLayout(self.image_navigation_layout)

        layout.addLayout(self.image_viewer_layout)
        self.image_viewer_tab.setLayout(layout)

    def init_dataset_stats_tab(self):
        layout = QVBoxLayout()

        self.stats_layout = QVBoxLayout()
        layout.addLayout(self.stats_layout)

        self.dataset_stats_tab.setLayout(layout)


    def generate_class_colors(self):
        def random_color():
            return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for class_id in self.class_colors:
            self.class_colors[class_id] = random_color()
        self.update_class_colors_table()

    def on_color_cell_clicked(self, item):
        if item.column() == 1:  # Check if the clicked cell is in the color column
            row = item.row()
            class_id = self.class_colors_table.item(row, 0).text()
            color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
            if color.isValid():
                self.class_colors[class_id] = color
                self.update_class_colors_table()
                self.show_image()

    def update_class_colors_table(self):
        self.class_colors_table.setRowCount(len(self.class_colors))
        for row, (class_id, color) in enumerate(self.class_colors.items()):
            
            class_item = QTableWidgetItem(class_id)
            color_item = QTableWidgetItem()
            color_item.setBackground(color)
            self.class_colors_table.setItem(row, 0, class_item)
            self.class_colors_table.setItem(row, 1, color_item)

    def change_class_color(self):
        selected_items = self.class_colors_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a class to change its color.")
            return

        class_item = selected_items[0]
        class_id = class_item.text()

        color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
        if color.isValid():
            self.class_colors[class_id] = color
            self.update_class_colors_table()
            self.show_image()

    def get_dataset_stats(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select the dataset root.")
            return

        self.clear_layout(self.stats_layout)

        class_counter = Counter()
        image_counter = 0
        instance_counter = 0

        for label_path in self.label_paths.values():
            if os.path.exists(label_path):
                image_counter += 1
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        class_id = line.strip().split()[0]
                        class_counter[class_id] += 1
                        instance_counter += 1

        total_classes = len(class_counter)
        total_instances = instance_counter
        avg_instances_per_image = total_instances / image_counter if image_counter else 0

        stats_label = QLabel(f"Total Classes: {total_classes}")
        self.stats_layout.addWidget(stats_label)

        images_label = QLabel(f"Total Images: {image_counter}")
        self.stats_layout.addWidget(images_label)

        instances_label = QLabel(f"Total Instances: {total_instances}")
        self.stats_layout.addWidget(instances_label)

        avg_instances_label = QLabel(f"Average Instances per Image: {avg_instances_per_image:.2f}")
        self.stats_layout.addWidget(avg_instances_label)

        # Create a list of class names, using YAML labels if available
        class_names = []
        for class_id in class_counter.keys():
            if hasattr(self, 'yaml_labels') and self.yaml_labels:
                try:
                    # Try to convert numeric class_id to YAML label
                    label = self.yaml_labels[int(class_id)] if int(class_id) < len(self.yaml_labels) else class_id
                    class_names.append(label)
                except (ValueError, IndexError):
                    class_names.append(class_id)
            else:
                class_names.append(class_id)

        class_table = QTableWidget()
        class_table.setColumnCount(3)
        class_table.setHorizontalHeaderLabels(["Class", "Instances", "Percentage"])
        class_table.setRowCount(len(class_counter))
        class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, (class_id, class_name) in enumerate(zip(class_counter.keys(), class_names)):
            count = class_counter[class_id]
            class_item = QTableWidgetItem(str(class_name))
            count_item = QTableWidgetItem(str(count))
            percentage_item = QTableWidgetItem(f"{(count / total_instances) * 100:.2f}%")
            class_table.setItem(row, 0, class_item)
            class_table.setItem(row, 1, count_item)
            class_table.setItem(row, 2, percentage_item)

        self.stats_layout.addWidget(class_table)

        # Plotting a bar graph for class distribution
        fig, ax = plt.subplots()
        
        # Use YAML labels or fall back to numeric class IDs
        classes = class_names
        counts = list(class_counter.values())

        # Assign colors to classes
        self.class_colors = {str(class_name): QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for class_name in classes}
        colors = [self.class_colors[str(class_name)] for class_name in classes]

        bars = ax.bar(classes, counts, color=[self.rgb_to_hex(c) for c in colors])
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Instances')
        ax.set_title('Class Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Add text labels above bars
        for bar, count in zip(bars, counts):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, int(count), 
                    ha='center', va='bottom')

        canvas = FigureCanvas(fig)
        self.stats_layout.addWidget(canvas)

        self.update_class_colors_table()  # Populate the colors table after generating dataset stats

    def rgb_to_hex(self, qcolor):
        return '#{:02x}{:02x}{:02x}'.format(qcolor.red(), qcolor.green(), qcolor.blue())

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

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

    def add_slider_to_layout(self, layout, label_text, slider, value_label, row):
        label = QLabel(label_text)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

    def select_dataset_root(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if dir_name:
            self.dataset_root = dir_name
            self.dataset_label.setText(dir_name)
            if not self.output_dir_set:
                self.prompt_for_output_dir()
            self.update_sliders_state()
            self.scan_folders()
            self.get_dataset_stats()  # Generate dataset stats when dataset is loaded

    def select_overlay_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Overlay Image Directory")
        if dir_name:
            self.overlay_image_dir = dir_name
            self.overlay_label.setText(dir_name)
            self.update_sliders_state()

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_dir = dir_name
            self.output_dir_label.setText(dir_name)
        else:
            self.prompt_for_output_dir()

    def prompt_for_output_dir(self):
        while not self.output_dir:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Output Directory")
            msg_box.setText(f"Would you like to specify an output directory? \n Default: {self.dataset_root}_Augmented")
            specify_btn = msg_box.addButton("Specify", QMessageBox.ButtonRole.AcceptRole)
            default_btn = msg_box.addButton("Default", QMessageBox.ButtonRole.RejectRole)
            msg_box.exec()

            if msg_box.clickedButton() == specify_btn:
                dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if dir_name:
                    self.output_dir = dir_name
                    self.output_dir_label.setText(dir_name)
            else:
                self.output_dir = self.dataset_root + "_Augmented"
                self.output_dir_label.setText(self.output_dir)
            self.output_dir_set = True

    def scan_folders(self):
        # Clear previous skip augmentation inputs
        for key in self.skip_augmentations.keys():
            self.skip_augmentations[key] = []

        # Scan dataset for folders and images
        folders = set()
        self.image_paths = []  # Reset image paths
        self.label_paths = {}  # Reset label paths

        for root, dirs, files in os.walk(self.dataset_root):
            if os.path.basename(root).lower() not in ['images', 'labels']:
                for name in dirs:
                    if name.lower() not in ['train', 'val', 'labels', 'images']:
                        folders.add(name)
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        self.image_paths.append(image_path)
                        label_path = os.path.join(self.dataset_root, 'labels', os.path.relpath(image_path, os.path.join(self.dataset_root, 'images')).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                        self.label_paths[image_path] = label_path

        folders = list(folders)
        folders.sort()

        self.skip_table.setRowCount(len(folders))

        self.folder_list.clear()
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make folder names read-only
            self.skip_table.setItem(row, 0, folder_item)
            list_item = QListWidgetItem(folder)
            self.folder_list.addItem(list_item)
            for col in range(1, 7):  # Update the range to include the new column
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 0px; margin-right: auto;")  # Align checkbox to the left 
                if col == 5:
                    checkbox.setEnabled(False)
                if col == 6:  # Connect the new checkbox to the slot
                    checkbox.stateChanged.connect(lambda state, r=row: self.toggle_skip_all(state, r))
                self.skip_table.setCellWidget(row, col, checkbox)

        # Sort images numerically
        self.image_paths.sort(key=self.natural_keys)

        self.generate_class_colors()  # Generate class colors after loading dataset

        yaml_labels = parse_dataset_yaml(self.dataset_root)
    
        # Store the YAML labels as an instance attribute for later use
        self.yaml_labels = yaml_labels
        
        # If YAML is found, update class colors without losing existing color assignments
        if yaml_labels:
            # Create a mapping that preserves existing colors
            new_class_colors = {}
            for class_id, color in self.class_colors.items():
                # Convert int keys to strings to match existing class_colors format
                try:
                    # Try to convert numeric class_id to a name from YAML
                    label = yaml_labels[int(class_id)] if int(class_id) < len(yaml_labels) else class_id
                    new_class_colors[str(label)] = color
                except (ValueError, IndexError):
                    # If conversion fails, keep the original class_id
                    new_class_colors[class_id] = color
            
            self.class_colors = new_class_colors
            
            # Update table and visualization
            self.update_class_colors_table()
            self.show_image()  # Refresh the image with new labels
            
            # Update dataset stats graph to use YAML labels
            if hasattr(self, 'get_dataset_stats'):
                self.get_dataset_stats()

    def toggle_skip_all(self, state, row):
        skip_all_checked = state == Qt.CheckState.Checked
        for col in range(1, 6):  # Update to check relevant columns
            checkbox = self.skip_table.cellWidget(row, col)
            checkbox.setEnabled(not skip_all_checked)
        if not self.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            overlay_checkbox.setEnabled(False)

    def display_images(self, item):
        current_show_original = self.show_original  # Store the current state
        
        self.folder_name = item.text()
        self.folder_images = [
            path for path in self.image_paths 
            if os.path.basename(os.path.dirname(path)) == self.folder_name
        ]
        self.image_slider.setMaximum(len(self.folder_images) - 1)
        self.current_image_index = 0
        self.augmented_image = None
        
        # Restore the state before showing the image
        self.show_original = current_show_original
        self.show_image()

    
    def show_image(self):
        try:
            if self.augmented_image is not None:
                if not self.show_original:
                    self.image_name_label.setText(f"(Preview) {os.path.basename(self.current_image_path)}")
                    self.display_image_and_polygons(self.augmented_image, self.augmented_polygons)
                    self.show_original_btn.setText("Show Original Image")
                else:
                    self.image_name_label.setText(f"(Original) {os.path.basename(self.current_image_path)}")
                    original_image = cv2.imread(self.current_image_path)
                    polygons, _ = self.load_polygons_and_labels(self.label_paths.get(self.current_image_path), original_image.shape)
                    self.display_image_and_polygons(original_image, polygons)
                    self.show_original_btn.setText("Show Augmented Image")
            elif self.folder_images:
                self.current_image_path = self.folder_images[self.current_image_index]
                
                # Use image cache
                image = self.image_cache.get_image(self.current_image_path)
                if image is None:
                    return
                    
                # Check for augmented version
                relative_image_path = os.path.relpath(self.current_image_path, self.dataset_root)
                augmented_image_path = os.path.join(self.output_dir, relative_image_path)

                if os.path.exists(augmented_image_path):
                    if not self.show_original:
                        image = self.image_cache.get_image(augmented_image_path)
                        self.image_name_label.setText(f"(Augmented) {os.path.basename(self.current_image_path)}")
                        self.show_original_btn.setText("Show Original Image")
                    else:
                        self.image_name_label.setText(f"(Original) {os.path.basename(self.current_image_path)}")
                        self.show_original_btn.setText("Show Augmented Image")
                    self.show_original_btn.setEnabled(True)
                else:
                    self.image_name_label.setText(os.path.basename(self.current_image_path))
                    self.show_original_btn.setText("Show Original Image")
                    self.show_original_btn.setEnabled(False)
                    self.show_original = False  # Reset state if no augmented version exists

                # Process labels
                label_path = self.label_paths.get(self.current_image_path)
                if self.show_original:
                    polygons, labels = self.load_polygons_and_labels(label_path, image.shape)
                else:
                    relative_label_path = os.path.relpath(label_path, self.dataset_root)
                    augmented_label_path = os.path.join(self.output_dir, relative_label_path)
                    if os.path.exists(augmented_label_path):
                        polygons, labels = self.load_polygons_and_labels(augmented_label_path, image.shape)
                    else:
                        polygons, labels = self.load_polygons_and_labels(label_path, image.shape)

                self.display_image_and_polygons(image, polygons)
                self.update_navigation_buttons()
                
        except Exception as e:
            print(f"Error showing image: {str(e)}")

    def show_original_image(self):
        if self.current_image_path:
            self.show_original = not self.show_original  # Toggle the state
            self.show_image()  # Refresh the display with the new state


    def update_sliders_state(self):
        enable_normal_sliders = bool(self.dataset_root)
        enable_overlay_sliders = bool(self.overlay_image_dir)
        
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
            overlay_checkbox.setEnabled(enable_overlay_sliders)

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

    def toggle_bounding_boxes(self):
        self.show_bounding_boxes = self.bbox_checkbox.isChecked()
        self.show_image()

    def toggle_polygons(self):
        self.show_polygons = self.polygons_checkbox.isChecked()
        self.show_image()

    def toggle_labels(self):
        self.show_labels = self.labels_checkbox.isChecked()
        self.show_image()

    def toggle_points(self):
        self.show_points = self.points_checkbox.isChecked()
        self.show_image()

    def display_image_and_polygons(self, image, polygons):
        print("Displaying image with class colors:", self.class_colors)
        
        for polygon in polygons:
            class_id = polygon[0]
            print(f"Polygon class_id: {class_id}")
            print(f"Color for this class: {self.class_colors.get(class_id, 'Not found')}")
            
            if class_id not in self.class_colors:
                print(f"Class {class_id} not in class_colors, generating new color")
                self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        height, width, _ = image.shape
        image_bytes = image.tobytes()
        qimage = QImage(image_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        qimage = qimage.rgbSwapped()

        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(scaled_pixmap)

        # Enable anti-aliasing for sharper lines and text
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)


        labels = []  # To store labels and positions for later drawing

        if self.augmented_image is not None and self.augmented_image_original_dims is not None:
            orig_h, orig_w = self.augmented_image_original_dims
        else:
            orig_h, orig_w = height, width

        for polygon in polygons:
            class_id = polygon[0]
            if class_id not in self.class_colors:
                self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Set pen for polygon lines and bounding boxes to full opacity
            pen = QPen(self.class_colors[class_id], 2)
            pen_color = QColor(self.class_colors[class_id].red(), self.class_colors[class_id].green(), self.class_colors[class_id].blue(), 255) # Full opacity
            pen.setColor(pen_color)

            # Set brush color with desired opacity
            brush_color = QColor(self.class_colors[class_id].red(), self.class_colors[class_id].green(), self.class_colors[class_id].blue(), 100)  # Set fill opacity here (0-255)
            brush = QBrush(brush_color)
            brush.setStyle(Qt.BrushStyle.SolidPattern)

            points = [QPointF(pt[0] * scaled_pixmap.width() / orig_w, pt[1] * scaled_pixmap.height() / orig_h) for pt in polygon[1:]]

            if self.show_polygons:
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPolygon(*points)

            if self.show_points:
                for point in points:
                    painter.setPen(QPen(Qt.GlobalColor.black, 1))
                    painter.drawEllipse(point, 2.5, 2.5)
                    painter.setPen(QPen(pen_color, 1))
                    painter.drawEllipse(point, 1.5, 1.5)

            # Calculate bounding box
            min_x = min(point.x() for point in points)
            max_x = max(point.x() for point in points)
            min_y = min(point.y() for point in points)
            max_y = max(point.y() for point in points)

            if self.show_bounding_boxes:
                # Draw bounding box with full opacity
                bounding_box_pen = QPen(pen_color, 1)
                painter.setPen(bounding_box_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))

            # Store label information for later drawing
            labels.append((class_id, points[0]))

        # Draw all labels
        # Draw all labels
        if self.show_labels:
            for class_id, position in labels:
                # Check if we have a YAML label for this class
                display_label = class_id
                if hasattr(self, 'yaml_labels') and self.yaml_labels:
                    try:
                        # Try to convert numeric class_id to YAML label
                        display_label = self.yaml_labels[int(class_id)] if int(class_id) < len(self.yaml_labels) else class_id
                    except (ValueError, IndexError):
                        # If conversion fails, keep the original class_id
                        pass

                # Draw text with black outline
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                font_metrics = painter.fontMetrics()
                text_height = font_metrics.height()
                text_width = font_metrics.horizontalAdvance(display_label)
                
                # Adjust label position to prevent going off the canvas
                label_x = position.x()
                label_y = position.y() + text_height

                # Ensure X-coordinate is within canvas
                label_x = max(0, min(label_x, scaled_pixmap.width() - text_width))
                
                # Ensure Y-coordinate is within canvas
                label_y = max(text_height, min(label_y, scaled_pixmap.height() - 5))

                label_position = QPointF(label_x, label_y)
                
                # Draw black outline
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    painter.drawText(label_position + QPointF(dx, dy), display_label)

                # Draw text in white on top
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(label_position, display_label)

        painter.end()

        self.image_label.setPixmap(scaled_pixmap)

    def convert_bbox_to_polygon(self, bbox):
        class_id = bbox[0]
        x_center, y_center, width, height = map(float, bbox[1:])
        half_w = width / 2
        half_h = height / 2
        points = [
            x_center - half_w, y_center - half_h,
            x_center + half_w, y_center - half_h,
            x_center + half_w, y_center + half_h,
            x_center - half_w, y_center + half_h
        ]
        return [class_id] + points

    def load_polygons_and_labels(self, label_path, target_size):
        polygons = []

        if (label_path and os.path.exists(label_path)):
            with open(label_path, 'r') as f:
                label_data = f.readlines()

            for line in label_data:
                line_data = line.strip().split()
                annotation_type = self.identify_annotation_type(line_data)
                if (annotation_type == 'bbox'):
                    polygon_data = self.convert_bbox_to_polygon(line_data)
                else:
                    polygon_data = line_data

                if len(polygon_data) < 5:
                    continue  # Ensure there are enough coordinates for a polygon

                class_id = str(polygon_data[0])  # Convert to string here
                if class_id not in self.class_colors:
                    self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                pen = QPen(self.class_colors[class_id], 2)
                brush_color = QColor(self.class_colors[class_id].red(), self.class_colors[class_id].green(), self.class_colors[class_id].blue(), 100)  # Set opacity here (0-255)
                brush = QBrush(brush_color)
                brush.setStyle(Qt.BrushStyle.SolidPattern)

                # Extract normalized coordinates
                coords = list(map(float, polygon_data[1:]))
                points = [(coords[i] * target_size[1], coords[i+1] * target_size[0]) for i in range(0, len(coords), 2)]

                polygons.append([class_id] + points)

        return polygons, []

    def identify_annotation_type(self, parts):
        
        if len(parts) < 5:
            return "unknown"
        if len(parts) % 2 == 1 and len(parts) > 5:
            return "polygon"
        elif len(parts) == 5:
            return "bbox"
        else:
            return "unknown"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.folder_images) - 1)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_slider.setValue(self.current_image_index)
            self.augmented_image = None
            self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.folder_images) - 1:
            self.current_image_index += 1
            self.image_slider.setValue(self.current_image_index)
            self.augmented_image = None
            self.show_image()

    def slider_value_changed(self, value):
        if value != self.current_image_index:
            self.current_image_index = value
            self.augmented_image = None
            self.show_image()

    def run_augmentation(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select the dataset root.")
            return

        if not self.output_dir_set:
            self.prompt_for_output_dir()

        # Create progress dialog without cancel button
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setCancelButton(None)  # Remove the default cancel button
        self.progress_dialog.setWindowTitle("Processing Images")
        self.progress_dialog.setMinimumWidth(600)
        self.progress_dialog.setMinimumHeight(300)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        
        # Create layout for the progress dialog
        layout = QVBoxLayout(self.progress_dialog)
        
        # Info labels
        self.progress_label = QLabel("Starting...")
        self.time_label = QLabel("Estimated time remaining: Calculating...")
        layout.addWidget(self.progress_label)
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
        self.progress_dialog.findChild(QLabel).hide()
        
        # Make the dialog modal
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        
        # Prepare parameters (same as before)
        params = {
            'image_dir': os.path.join(self.dataset_root, 'images'),
            'label_dir': os.path.join(self.dataset_root, 'labels'),
            'augmented_image_dir': os.path.join(self.output_dir, 'images'),
            'augmented_label_dir': os.path.join(self.output_dir, 'labels'),
            'skip_existing': self.skip_existing_checkbox.isChecked(),
            'skip_augmentations': self.get_skip_augmentations(),
            'mirror_weights': [self.mirror_slider.value(), 100 - self.mirror_slider.value()],
            'crop_weights': [self.crop_slider.value(), 100 - self.crop_slider.value()],
            'zoom_weights': [self.zoom_slider.value(), 100 - self.zoom_slider.value()],
            'rotate_weights': [self.rotate_slider.value(), 100 - self.rotate_slider.value()],
            'overlay_weights': ([self.overlay_slider.value(), 100 - self.overlay_slider.value()] 
                            if self.overlay_image_dir else [0, 100]),
            'rotation_random_vs_90_weights': [self.rotation_random_vs_90_slider.value(), 
                                            100 - self.rotation_random_vs_90_slider.value()],
            'overlay_min_max_scale': self.overlay_min_max_scale,
            'maintain_aspect_ratio_weights': [self.maintain_aspect_ratio_slider.value(),
                                            100 - self.maintain_aspect_ratio_slider.value()],
            'zoom_in_vs_out_weights': [self.zoom_in_vs_out_slider.value(),
                                    100 - self.zoom_in_vs_out_slider.value()],
            'zoom_padding': self.zoom_padding,
            'coco_image_folder': self.overlay_image_dir if self.overlay_image_dir else ""
        }

        # Create and configure worker
        self.worker = AugmentationWorker(params)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.progress.connect(self.update_progress)  
        self.worker.progress_log.connect(lambda msg: self.log_text.append(msg))
        self.worker.finished.connect(self.handle_completion)
        self.worker.error.connect(self.handle_augmentation_error)
        
        # Initial time for ETA calculation
        self.start_time = time.time()
        self.is_cancelled = False
        
        # Start processing
        self.worker.start()
        self.progress_dialog.exec()  # Show the dialog and wait for completion

    def update_progress(self, value):
        """Update progress and time estimates"""
        if not self.is_cancelled and value > 0:
            try:
                elapsed_time = time.time() - self.start_time
                estimated_total_time = elapsed_time * 100 / value
                remaining_time = estimated_total_time - elapsed_time
                
                # Format time remaining
                if remaining_time < 60:
                    time_str = f"{int(remaining_time)} seconds"
                elif remaining_time < 3600:
                    time_str = f"{int(remaining_time / 60)} minutes"
                else:
                    time_str = f"{remaining_time / 3600:.1f} hours"
                
                self.time_label.setText(f"Estimated time remaining: {time_str}")
                self.progress_label.setText(f"Progress: {value}%")
            except RuntimeError:
                # Widget has been deleted, ignore the update
                pass
            
    def handle_completion(self):
        """Handle successful completion of the augmentation process"""
        if not self.is_cancelled:
            self.progress_dialog.close()
            QMessageBox.information(self, "Complete", "Augmentation process completed successfully!")

    def handle_cancellation(self):
        """Handle user cancellation of the augmentation process"""
        self.is_cancelled = True
        self.worker.cancel()
        self.log_text.append("\nCancelling...")
        self.progress_dialog.close()
        
    def handle_augmentation_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during augmentation: {error_msg}")

    def augment_current_image(self):
        if not self.dataset_root or not self.overlay_image_dir:
            overlay_weights = [0, 100]
        else:
            overlay_weights = [self.overlay_slider.value(), 100 - self.overlay_slider.value()]
        
        if not self.current_image_path:
            QMessageBox.warning(self, "Input Required", "Please select an image to augment.")
            return
        
        mirror_weights = [self.mirror_slider.value(), 100 - self.mirror_slider.value()]
        crop_weights = [self.crop_slider.value(), 100 - self.crop_slider.value()]
        zoom_weights = [self.zoom_slider.value(), 100 - self.zoom_slider.value()]
        rotate_weights = [self.rotate_slider.value(), 100 - self.rotate_slider.value()]
        #overlay_scale_weights = [self.overlay_scale_slider.value(), 100 - self.overlay_scale_slider.value()]
        maintain_aspect_ratio_weights = [self.maintain_aspect_ratio_slider.value(), 100 - self.maintain_aspect_ratio_slider.value()]
        zoom_in_vs_out_weights = [self.zoom_in_vs_out_slider.value(), 100 - self.zoom_in_vs_out_slider.value()]
        rotation_random_vs_90_weights = [self.rotation_random_vs_90_slider.value(), 100 - self.rotation_random_vs_90_slider.value()]

        # Load the image
        image = cv2.imread(self.current_image_path)
        (h, w) = image.shape[:2]

        # Load the label file if it exists
        label_path = self.label_paths.get(self.current_image_path)
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                polygons = []
                class_ids = []
                for line in lines:
                    parts = line.strip().split()
                    if self.identify_annotation_type(parts) == 'bbox':
                        parts = self.convert_bbox_to_polygon(parts)
                    
                    class_id = parts[0]
                    class_ids.append(class_id)
                    polygon = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
                    polygons.append(polygon)
        else:
            polygons = []
            class_ids = []

        # Run the augment_image function
        augmented_image, augmented_polygons = augment_image(
            image,
            polygons,
            self.folder_name,
            class_ids,
            h,
            w,
            self.skip_augmentations, 
            mirror_weights, 
            crop_weights,
            overlay_weights, 
            rotate_weights,
            rotation_random_vs_90_weights,
            self.overlay_min_max_scale,
            maintain_aspect_ratio_weights, 
            zoom_weights, 
            zoom_in_vs_out_weights,
            self.zoom_padding,
            self.overlay_image_dir if self.overlay_image_dir else ""
        )

        (new_h, new_w) = augmented_image.shape[:2]

        denormalized_polygons = []
        for polygon in augmented_polygons:
            class_id = polygon[0]
            denormalized_polygon = [class_id] + [(int(x * new_w), int(y * new_h)) for (x, y) in polygon[1:]]
            denormalized_polygons.append(denormalized_polygon)

        self.augmented_image = augmented_image
        self.augmented_polygons = denormalized_polygons
        self.augmented_image_original_dims = (new_h, new_w)
        self.show_image()

    def augment_and_save_current_image(self):
        self.augment_current_image()
        self.save_augmented_image()

    def save_current_preview(self):
        if self.augmented_image is None or not self.augmented_image.any():
            QMessageBox.warning(self, "No Augmented Image", "There is no augmented image to save.")
            return
        
        # Always force overwrite when saving preview
        self.save_augmented_image(force_overwrite=True)

    def save_augmented_image(self, force_overwrite=False):
        if not self.output_dir_set:
            self.prompt_for_output_dir()

        if self.augmented_image is not None:
            relative_image_path = os.path.relpath(self.current_image_path, self.dataset_root)
            augmented_image_path = os.path.join(self.output_dir, relative_image_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)

            # Always overwrite if force_overwrite is True
            if force_overwrite:
                result = cv2.imwrite(augmented_image_path, self.augmented_image)
                if not result:
                    QMessageBox.warning(self, "Save Error", f"Failed to save image at {augmented_image_path}")
            elif not os.path.exists(augmented_image_path):
                result = cv2.imwrite(augmented_image_path, self.augmented_image)
                if not result:
                    QMessageBox.warning(self, "Save Error", f"Failed to save image at {augmented_image_path}")
            else:
                QMessageBox.information(self, "Skip", f"Image already exists and was skipped: {augmented_image_path}")

            # Save augmented label
            relative_label_path = os.path.relpath(self.label_paths[self.current_image_path], self.dataset_root)
            augmented_label_path = os.path.join(self.output_dir, relative_label_path)

            os.makedirs(os.path.dirname(augmented_label_path), exist_ok=True)
            with open(augmented_label_path, 'w') as f:
                for polygon in self.augmented_polygons:
                    class_id = polygon[0]
                    coords = [f"{x / self.augmented_image_original_dims[1]} {y / self.augmented_image_original_dims[0]}" for x, y in polygon[1:]]
                    f.write(f"{class_id} {' '.join(coords)}\n")

            # Invalidate cache to force reload
            if augmented_image_path in self.image_cache.cache:
                del self.image_cache.cache[augmented_image_path]



    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AugmentDatasetGUI()
    ex.show()
    sys.exit(app.exec())