from PyQt6.QtWidgets import (QGroupBox, QFormLayout, QLabel, QPushButton, QFileDialog)

class DirectorySelector(QGroupBox):
    """Component for selecting dataset directories"""
    
    def __init__(self, parent):
        super().__init__("Select Directories")
        self.parent = parent
        self.initUI()
        
    def initUI(self):
        dir_layout = QFormLayout()
        
        # Create labels for displaying selected paths
        self.dataset_label = QLabel("Not selected")
        self.overlay_label = QLabel("Not selected")
        self.output_dir_label = QLabel("Not selected")
        
        # Create buttons for directory selection
        self.dataset_btn = QPushButton("Select Dataset Root")
        self.overlay_btn = QPushButton("Select Overlay Image Directory")
        self.output_dir_btn = QPushButton("Select Output Directory")
        
        # Connect button signals to slots
        self.dataset_btn.clicked.connect(self.select_dataset_root)
        self.overlay_btn.clicked.connect(self.select_overlay_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        
        # Add components to layout
        dir_layout.addRow(self.dataset_btn, self.dataset_label)
        dir_layout.addRow(self.overlay_btn, self.overlay_label)
        dir_layout.addRow(self.output_dir_btn, self.output_dir_label)
        
        self.setLayout(dir_layout)
        
    def select_dataset_root(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if dir_name:
            self.dataset_label.setText(dir_name)
            self.parent.parent.select_dataset_root(dir_name)
            
    def select_overlay_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Overlay Image Directory")
        if dir_name:
            self.overlay_label.setText(dir_name)
            self.parent.parent.select_overlay_dir(dir_name)
            
    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_dir_label.setText(dir_name)
            self.parent.select_output_dir(dir_name)
    
    def get_paths(self):
        """Return the currently selected directory paths"""
        return {
            "dataset": self.dataset_label.text() if self.dataset_label.text() != "Not selected" else None,
            "overlay": self.overlay_label.text() if self.overlay_label.text() != "Not selected" else None,
            "output": self.output_dir_label.text() if self.output_dir_label.text() != "Not selected" else None
        }