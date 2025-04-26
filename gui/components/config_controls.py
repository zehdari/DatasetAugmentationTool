from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QCheckBox)

class ConfigControls(QWidget):
    """Component for config loading/saving and global settings"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Skip existing checkbox
        self.skip_existing_checkbox = QCheckBox("Skip Already Augmented Images")
        self.skip_existing_checkbox.setChecked(True)
        layout.addWidget(self.skip_existing_checkbox)
        
        # Config buttons
        self.load_config_button = QPushButton("Load Config")
        self.save_config_button = QPushButton("Save Config")
        
        self.load_config_button.clicked.connect(self.load_config)
        self.save_config_button.clicked.connect(self.save_config)
        
        layout.addWidget(self.load_config_button)
        layout.addWidget(self.save_config_button)
        
        self.setLayout(layout)
        
    def load_config(self):
        """Load configuration from file"""
        config_data = self.parent.parent.config_manager.load_config()
        if config_data:
            # Notify parent to apply the loaded config
            self.parent.apply_config(config_data)
            
    def save_config(self):
        """Save current configuration to file"""
        # Get current config data from parent
        config_data = self.parent.get_config_data()
        self.parent.parent.config_manager.save_config(config_data)
    
    def is_skip_existing(self):
        """Return state of skip existing checkbox"""
        return self.skip_existing_checkbox.isChecked()