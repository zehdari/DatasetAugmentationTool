import json
from PyQt6.QtWidgets import QFileDialog, QMessageBox

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
                return config_data
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to load configuration: {e}")
        return None