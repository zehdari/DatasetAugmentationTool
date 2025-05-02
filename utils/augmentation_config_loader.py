# augmentation_config_loader.py
import yaml
import os
from PyQt6.QtGui import QColor
import random

class AugmentationConfigLoader:
    """Loads augmentation configurations from a YAML file."""
    
    def __init__(self, config_path='augmentation_config.yaml'):
        self.config_path = config_path
        self.augmentations = {}
        self.skip_categories = {}
        self.float_params = {}
        self.child_sliders = {}
        
    def load_config(self):
        """Load the augmentation configuration from YAML."""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self._set_defaults()
            return False
            
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Process augmentations
            if 'augmentations' in config:
                self.augmentations = config['augmentations']
                
            # Process skip categories
            if 'skip_categories' in config:
                self.skip_categories = config['skip_categories']
                
            # Process float parameters
            if 'float_params' in config:
                self.float_params = config['float_params']
                
            # Process child sliders
            if 'child_sliders' in config:
                self.child_sliders = config['child_sliders']
                
            return True
                
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self._set_defaults()
            return False
    
    def _set_defaults(self):
        """Set default configurations if the YAML file is not found or has errors."""
        # Default augmentations
        self.augmentations = {
            "mirror": {
                "name": "Mirror % Probability:",
                "slider_attr": "mirror_slider",
                "value_attr": "mirror_value",
                "default_value": 50,
                "details": "Mirrors the image horizontally (left to right). This creates a flipped version of the original image. The percentage controls how often mirroring is applied."
            },
            "rotate": {
                "name": "Rotate % Probability:",
                "slider_attr": "rotate_slider",
                "value_attr": "rotate_value",
                "default_value": 50,
                "details": "Rotates the image. Higher percentage means rotation will be applied more frequently."
            },
            "crop": {
                "name": "Crop % Probability:",
                "slider_attr": "crop_slider",
                "value_attr": "crop_value",
                "default_value": 50,
                "details": "Crops a portion of the image. Higher percentage means cropping will be applied more frequently."
            },
            "zoom": {
                "name": "Zoom % Probability:",
                "slider_attr": "zoom_slider",
                "value_attr": "zoom_value",
                "default_value": 50,
                "details": "Zooms in or out of the image. Higher percentage means zoom operations will be applied more frequently."
            },
            "overlay": {
                "name": "Overlay % Probability:",
                "slider_attr": "overlay_slider",
                "value_attr": "overlay_value",
                "default_value": 50,
                "details": "Overlays objects from one image onto another. Higher percentage means overlays will be applied more frequently. Requires overlay directory selection."
            }
        }
        
        # Default skip categories
        self.skip_categories = ["Zoom", "Crop", "Rotate", "Mirror", "Overlay"]
        
        # Default float parameters
        self.float_params = {
            "zoom_slider": [
                {"attr": "zoom_in_min_padding", "name": "Zoom In Min Padding:", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                {"attr": "zoom_in_max_padding", "name": "Zoom In Max Padding:", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                {"attr": "zoom_out_min_padding", "name": "Zoom Out Min Padding:", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                {"attr": "zoom_out_max_padding", "name": "Zoom Out Max Padding:", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}
            ],
            "overlay_slider": [
                {"attr": "overlay_min_scale", "name": "Overlay Min Scale:", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                {"attr": "overlay_max_scale", "name": "Overlay Max Scale:", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}
            ]
        }
        
        # Default child sliders
        self.child_sliders = {
            "rotate_slider": [
                {"attr": "rotation_random_vs_90_slider", "name": "Rotation (0 to 360) vs 90 %: ", "default": 25}
            ],
            "crop_slider": [
                {"attr": "maintain_aspect_ratio_slider", "name": "Maintain Aspect Ratio on Crop %: ", "default": 50}
            ],
            "zoom_slider": [
                {"attr": "zoom_in_vs_out_slider", "name": "Zoom In vs Out %: ", "default": 40}
            ]
        }
    
    def get_augmentations(self):
        """Get the loaded augmentations."""
        return self.augmentations
        
    def get_skip_categories(self):
        """Get the skip categories for folders."""
        return self.skip_categories
        
    def get_float_params(self):
        """Get the float parameters for sliders."""
        return self.float_params
        
    def get_child_sliders(self):
        """Get the child sliders for parent sliders."""
        return self.child_sliders