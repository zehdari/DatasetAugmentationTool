from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QCheckBox, QAbstractItemView)
from PyQt6.QtCore import Qt

class SkipAugmentationTable(QGroupBox):
    """Component for managing which folders to skip for specific augmentations"""
    
    def __init__(self, parent):
        super().__init__("Skip Augmentations for Folders")
        self.parent = parent
        
        # Initialize empty skip data
        self.skip_augmentations = {
            'Zoom': [],
            'Crop': [],
            'Rotate': [],
            'Mirror': [],
            'Overlay': []
        }
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # Create table widget
        self.skip_table = QTableWidget()
        self.skip_table.setColumnCount(7)  
        self.skip_table.setHorizontalHeaderLabels(['Folder', 'Zoom', 'Crop', 'Rotate', 'Mirror', 'Overlay', 'Skip All'])
        
        # Configure column sizing
        self.skip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.skip_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        
        for col in range(1, 7):
            self.skip_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.skip_table.setColumnWidth(col, 50)
            
        layout.addWidget(self.skip_table)
        self.setLayout(layout)
    
    def update_folders(self, folders, has_overlay=False):
        """Update the table with the provided folder list"""
        self.skip_table.setRowCount(len(folders))
        
        # Add each folder as a row
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make folder names read-only
            self.skip_table.setItem(row, 0, folder_item)
            
            # Add checkboxes for each augmentation type
            for col in range(1, 7):  # Includes the skip all column
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 0px; margin-right: auto;")  # Align checkbox to the left 
                
                # Disable overlay checkbox if no overlay directory is selected
                if col == 5:  # Overlay column
                    checkbox.setEnabled(has_overlay)
                    
                # Connect the "Skip All" checkbox to enable/disable others
                if col == 6:  # Skip All column
                    checkbox.stateChanged.connect(lambda state, r=row: self.toggle_skip_all(state, r))
                    
                self.skip_table.setCellWidget(row, col, checkbox)

    def toggle_skip_all(self, state, row):
        """Toggle all checkboxes in a row when Skip All is checked"""
        skip_all_checked = state == Qt.CheckState.Checked
        
        # Update individual augmentation checkboxes
        for col in range(1, 6):  # Columns 1-5 are individual augmentations
            checkbox = self.skip_table.cellWidget(row, col)
            checkbox.setEnabled(not skip_all_checked)
            
        # Always disable overlay checkbox if no overlay directory
        if not self.parent.parent.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, 5)  # Overlay column
            overlay_checkbox.setEnabled(False)
    
    def get_skip_augmentations(self):
        """Get the current skip augmentation settings"""
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
        
    def update_overlay_state(self, has_overlay):
        """Update the state of overlay checkboxes based on overlay directory selection"""
        for row in range(self.skip_table.rowCount()):
            overlay_checkbox = self.skip_table.cellWidget(row, 5)  # Overlay column
            skip_all_checkbox = self.skip_table.cellWidget(row, 6)  # Skip All column
            
            # Only enable overlay checkbox if overlay dir exists and Skip All is not checked
            overlay_checkbox.setEnabled(has_overlay and not skip_all_checkbox.isChecked())