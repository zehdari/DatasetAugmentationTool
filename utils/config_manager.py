import yaml
from PyQt6.QtWidgets import QFileDialog, QMessageBox

class ConfigManager:
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file

    def save_config(self, config_data):
        """Save user configuration to a YAML file with a file dialog"""
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Configuration", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
                QMessageBox.information(None, "Success", f"Configuration saved to {file_path}")
                return True
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to save configuration: {e}")
        return False

    def load_config(self):
        """Load user configuration from a YAML file with a file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(None, "Load Configuration", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    config_data = yaml.safe_load(file)
                return config_data
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load configuration: {e}")
        return None