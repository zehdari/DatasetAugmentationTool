import os
import yaml

class DatasetManager:

    @staticmethod
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