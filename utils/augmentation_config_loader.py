import yaml
import os

class AugmentationConfigLoader:
    def __init__(self, config_path='config/augmentation_config.yaml'):
        self.config_path = config_path
        self.config = {}
        self.default_config = self._get_default_config()
        
    def load_config(self):
        """Load the configuration from a YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                # If config file doesn't exist, create directory and use defaults
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                self.config = self.default_config
                # Save default config
                self.save_config(self.config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self.config = self.default_config
        return self.config
    
    def save_config(self, config_data):
        """Save configuration to a YAML file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_augmentations(self):
        """Get the augmentation configurations"""
        if not self.config:
            self.load_config()
        return self.config.get('augmentations', self.default_config['augmentations'])
    
    def get_child_sliders(self):
        """Get the child slider configurations"""
        if not self.config:
            self.load_config()
        return self.config.get('child_sliders', self.default_config['child_sliders'])
    
    def get_float_params(self):
        """Get the float parameter configurations"""
        if not self.config:
            self.load_config()
        return self.config.get('float_params', self.default_config['float_params'])
    
    def get_skip_categories(self):
        """Get the skip categories"""
        if not self.config:
            self.load_config()
        return self.config.get('skip_categories', self.default_config['skip_categories'])

    def _get_default_config(self):
        """Get default configuration settings"""
        return {
            'augmentations': {
                'mirror': {
                    'name': 'Mirror (Horizontal Flip)',
                    'slider_attr': 'mirror_slider',
                    'value_attr': 'mirror_value',
                    'default_value': 50,
                    'details': 'Horizontally flips the image with the specified probability.'
                },
                'crop': {
                    'name': 'Crop',
                    'slider_attr': 'crop_slider',
                    'value_attr': 'crop_value',
                    'default_value': 25,
                    'details': 'Crops a portion of the image with the specified probability.'
                },
                'zoom': {
                    'name': 'Zoom',
                    'slider_attr': 'zoom_slider',
                    'value_attr': 'zoom_value',
                    'default_value': 35,
                    'details': 'Zooms in or out of the image with the specified probability.'
                },
                'rotate': {
                    'name': 'Rotate',
                    'slider_attr': 'rotate_slider',
                    'value_attr': 'rotate_value',
                    'default_value': 30,
                    'details': 'Rotates the image with the specified probability.'
                },
                'overlay': {
                    'name': 'Overlay',
                    'slider_attr': 'overlay_slider',
                    'value_attr': 'overlay_value',
                    'default_value': 20,
                    'details': 'Overlays the detections on COCO images with the specified probability.'
                }
            },
            'child_sliders': {
                'rotate_slider': [
                    {
                        'name': 'Random vs. 90° Rotation',
                        'attr': 'rotation_random_vs_90_slider',
                        'default': 25
                    }
                ],
                'zoom_slider': [
                    {
                        'name': 'Zoom In vs. Out',
                        'attr': 'zoom_in_vs_out_slider',
                        'default': 70
                    }
                ],
                'crop_slider': [
                    {
                        'name': 'Maintain Aspect Ratio',
                        'attr': 'maintain_aspect_ratio_slider',
                        'default': 80
                    }
                ]
            },
            'float_params': {
                'zoom_slider': [
                    {
                        'name': 'Zoom In Min Padding',
                        'attr': 'zoom_in_min_padding',
                        'default': 0.1,
                        'min': 0.0,
                        'max': 0.5,
                        'step': 0.05
                    },
                    {
                        'name': 'Zoom In Max Padding',
                        'attr': 'zoom_in_max_padding',
                        'default': 0.3,
                        'min': 0.1,
                        'max': 0.8,
                        'step': 0.05
                    },
                    {
                        'name': 'Zoom Out Min Padding',
                        'attr': 'zoom_out_min_padding',
                        'default': 0.1,
                        'min': 0.0,
                        'max': 0.5,
                        'step': 0.05
                    },
                    {
                        'name': 'Zoom Out Max Padding',
                        'attr': 'zoom_out_max_padding',
                        'default': 0.5,
                        'min': 0.1,
                        'max': 1.0,
                        'step': 0.05
                    }
                ],
                'overlay_slider': [
                    {
                        'name': 'Overlay Min Scale',
                        'attr': 'overlay_min_scale',
                        'default': 0.3,
                        'min': 0.1,
                        'max': 0.9,
                        'step': 0.05
                    },
                    {
                        'name': 'Overlay Max Scale',
                        'attr': 'overlay_max_scale',
                        'default': 1.0,
                        'min': 0.2,
                        'max': 2.0,
                        'step': 0.05
                    }
                ]
            },
            'skip_categories': ['Mirror', 'Crop', 'Zoom', 'Rotate', 'Overlay']
        }