from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                             QProgressDialog, QSizePolicy)
from PyQt6.QtCore import Qt
import os
import re
import time

from utils.ui_components import ClickFilter
from utils.dataset_manager import DatasetManager

# Import our new components
from gui.components.directory_selector import DirectorySelector
from gui.components.slider_list import AugmentationSliderList
from gui.components.skip_table import SkipAugmentationTable
from gui.components.class_color_table import ClassColorTable
from gui.components.config_controls import ConfigControls
from gui.components.progress_window import AugmentationProgress

class AugmentationSettingsTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        # Initialize event filter
        self.installEventFilter(ClickFilter(self))
        
        # Initialize UI
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Directory selector component
        self.directory_selector = DirectorySelector(self)
        self.directory_selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.directory_selector)
        
        # Main splitter for sliders and tables
        weights_skip_layout = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Sliders and config controls
        self.slider_panel = QWidget()
        slider_layout = QVBoxLayout(self.slider_panel)
        
        # Augmentation sliders component
        self.sliders = AugmentationSliderList(self)
        slider_layout.addWidget(self.sliders)
        
        # Config controls component
        self.config_controls = ConfigControls(self)
        slider_layout.addWidget(self.config_controls)
        
        # Right panel: Skip augmentations and class colors
        self.tables_panel = QSplitter(Qt.Orientation.Vertical)
        
        # Skip augmentations component
        self.skip_table = SkipAugmentationTable(self)
        
        # Class colors component
        self.class_colors = ClassColorTable(self)
        
        # Add components to the vertical splitter
        self.tables_panel.addWidget(self.skip_table)
        self.tables_panel.addWidget(self.class_colors)
        self.tables_panel.setCollapsible(0, False)
        self.tables_panel.setCollapsible(1, False)
        
        # Add panels to horizontal splitter
        weights_skip_layout.addWidget(self.slider_panel)
        weights_skip_layout.addWidget(self.tables_panel)
        weights_skip_layout.setSizes([800, 400])  # Initial sizes of the panels
        weights_skip_layout.setCollapsible(0, False)
        weights_skip_layout.setCollapsible(1, False)
        
        # Set minimum sizes for panels
        self.tables_panel.setMinimumWidth(400)
        weights_skip_layout.setMinimumWidth(1200)
        weights_skip_layout.setHandleWidth(10)
        
        # Add the splitter to the main layout
        layout.addWidget(weights_skip_layout)
        
        # Set the main layout
        self.setLayout(layout)
        
    # Methods called by parent
    def select_dataset_root(self, dir_name):
        """Handle dataset root directory selection"""
        self.parent.select_dataset_root(dir_name)
        self.update_components_state()
        self.scan_folders()
        
    def select_overlay_dir(self, dir_name):
        """Handle overlay directory selection"""
        self.parent.select_overlay_dir(dir_name)
        self.update_components_state()
        
    def select_output_dir(self, dir_name):
        """Handle output directory selection"""
        self.parent.select_output_dir(dir_name)
        
    def update_components_state(self):
        """Update UI components based on selected directories"""
        has_dataset = bool(self.parent.dataset_root)
        has_overlay = bool(self.parent.overlay_image_dir)
        
        # Update slider states
        self.sliders.update_sliders_state(self.parent.dataset_root, self.parent.overlay_image_dir)
        
        # Update overlay checkboxes in skip table
        self.skip_table.update_overlay_state(has_overlay)
    
    def scan_folders(self):
        """Scan dataset for folders and update UI"""
        dataset_root = self.parent.dataset_root
        if not dataset_root:
            return
            
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
        
        # Update the skip table with folders
        self.skip_table.update_folders(folders, bool(self.parent.overlay_image_dir))
        
        # Keep a reference to the folder list in the image viewer if it exists
        if hasattr(self.parent, 'image_viewer_tab'):
            self.parent.image_viewer_tab.folder_list.clear()
            for folder in folders:
                self.parent.image_viewer_tab.folder_list.addItem(folder)

        # Sort images numerically
        image_paths.sort(key=self.natural_keys)
        
        # Update the parent's image paths
        self.parent.image_paths = image_paths
        self.parent.label_paths = label_paths

        # Parse YAML labels if available
        yaml_labels = DatasetManager.parse_dataset_yaml(dataset_root)
        
        # Update class colors with new labels
        self.class_colors.update_colors(yaml_labels)
        
        # Update the stats tab with the labels if it exists
        if hasattr(self.parent, 'stats_tab'):
            self.parent.stats_tab.yaml_labels = yaml_labels
            
    # Methods for augmentation
    def get_augmentation_params(self):
        """Get all parameters needed for augmentation"""
        # Get parameters from individual components
        params = self.sliders.get_augmentation_params()
        
        # Add skip existing setting
        params['skip_existing'] = self.config_controls.is_skip_existing()
        
        # Add skip augmentations
        params['skip_augmentations'] = self.skip_table.get_skip_augmentations()
        
        # If no overlay dir, set overlay weights to 0
        if not self.parent.overlay_image_dir:
            params['overlay_weights'] = [0, 100]
            
        return params
        
    def setup_progress_dialog(self, progress_dialog):
        """Set up the progress dialog for augmentation process"""
        self.progress_component = AugmentationProgress(self, progress_dialog)
        self.progress_component.start_tracking()
        
    def update_progress(self, value):
        """Update progress dialog"""
        if hasattr(self, 'progress_component'):
            self.progress_component.update_progress(value)
    
    # Configuration methods
    def apply_config(self, config_data):
        """Apply loaded configuration data"""
        if not config_data:
            return
            
        # Set slider values
        if "crop_probability" in config_data:
            self.sliders.crop_slider.setValue(config_data["crop_probability"])
        if "maintain_aspect_ratio" in config_data:
            self.sliders.maintain_aspect_ratio_slider.setValue(config_data["maintain_aspect_ratio"])
        if "mirror_probability" in config_data:
            self.sliders.mirror_slider.setValue(config_data["mirror_probability"])
        if "overlay_probability" in config_data:
            self.sliders.overlay_slider.setValue(config_data["overlay_probability"])
        if "rotate_probability" in config_data:
            self.sliders.rotate_slider.setValue(config_data["rotate_probability"])
        if "rotation_random_vs_90" in config_data:
            self.sliders.rotation_random_vs_90_slider.setValue(config_data["rotation_random_vs_90"])
        if "zoom_in_vs_out" in config_data:
            self.sliders.zoom_in_vs_out_slider.setValue(config_data["zoom_in_vs_out"])
        if "zoom_probability" in config_data:
            self.sliders.zoom_slider.setValue(config_data["zoom_probability"])
            
        # Set checkbox
        if "skip_existing" in config_data:
            self.config_controls.skip_existing_checkbox.setChecked(config_data["skip_existing"])
            
        # Apply saved order if available
        if "augmentation_order" in config_data:
            self.sliders.reorder_sliders_from_config(config_data["augmentation_order"])
            
        # Update slider states
        self.update_components_state()
        
    def get_config_data(self):
        """Get current configuration data for saving"""
        # Get the current order of sliders
        augmentation_order = self.sliders.get_augmentation_order()
        
        config_data = {
            "augmentation_order": augmentation_order,
            "crop_probability": self.sliders.crop_slider.value(),
            "maintain_aspect_ratio": self.sliders.maintain_aspect_ratio_slider.value(),
            "mirror_probability": self.sliders.mirror_slider.value(),
            "overlay_probability": self.sliders.overlay_slider.value(),
            "rotate_probability": self.sliders.rotate_slider.value(),
            "rotation_random_vs_90": self.sliders.rotation_random_vs_90_slider.value(),
            "zoom_in_vs_out": self.sliders.zoom_in_vs_out_slider.value(),
            "zoom_probability": self.sliders.zoom_slider.value(),
            "skip_existing": self.config_controls.is_skip_existing()
        }
        
        return config_data
    
    # Helper methods
    def atoi(self, text):
        """Helper function for natural sort order"""
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        """Sort text strings in natural order"""
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]