from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QMessageBox
from PyQt6.QtCore import Qt
from gui.settings_tab import AugmentationSettingsTab
from gui.image_viewer_tab import ImageViewerTab
from gui.stats_tab import DatasetStatsTab
from gui.progress_dialog import AugmentationProgressDialog
from services.augmentation_worker import AugmentationWorker
from utils.config_manager import ConfigManager
from utils.image_cache import ImageCache
import time
import os

class AugmentationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.image_cache = ImageCache(max_size=100)
        self.config_manager = ConfigManager('augmentation_config.json')
        
        self.dataset_root = ""
        self.overlay_image_dir = ""
        self.output_dir = ""
        self.output_dir_set = False
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Dataset Augmentation GUI')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        # Tabs
        self.tab_widget = QTabWidget()
        self.settings_tab = AugmentationSettingsTab(self)
        self.image_viewer_tab = ImageViewerTab(self)
        self.stats_tab = DatasetStatsTab(self)

        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.image_viewer_tab, "Image Viewer")
        self.tab_widget.addTab(self.stats_tab, "Dataset Stats")

        main_layout.addWidget(self.tab_widget)

        # Bottom button layout
        bottom_button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Augmentation")
        self.run_btn.clicked.connect(self.run_augmentation)
        bottom_button_layout.addWidget(self.run_btn)

        main_layout.addLayout(bottom_button_layout)
        self.setLayout(main_layout)

    def select_dataset_root(self, dir_name):
        if dir_name:
            self.dataset_root = dir_name
            if not self.output_dir_set:
                self.prompt_for_output_dir()
            self.settings_tab.update_sliders_state()
            self.settings_tab.scan_folders()
            self.stats_tab.get_dataset_stats()

    def select_overlay_dir(self, dir_name):
        if dir_name:
            self.overlay_image_dir = dir_name
            self.settings_tab.update_sliders_state()

    def select_output_dir(self, dir_name=None):
        if dir_name:
            self.output_dir = dir_name
            self.output_dir_set = True
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
                from PyQt6.QtWidgets import QFileDialog
                dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if dir_name:
                    self.output_dir = dir_name
                    # Update the label in the settings tab
                    self.settings_tab.output_dir_label.setText(dir_name)
            else:
                self.output_dir = self.dataset_root + "_Augmented"
                # Update the label in the settings tab with the default value
                self.settings_tab.output_dir_label.setText(self.output_dir)
            self.output_dir_set = True
            
    def run_augmentation(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select the dataset root.")
            return

        if not self.output_dir_set:
            self.prompt_for_output_dir()

        # Create progress dialog
        self.progress_dialog = AugmentationProgressDialog(self)
        
        # Connect progress dialog cancellation to handler
        self.progress_dialog.connect_cancel_button(self.handle_cancellation)
        
        # Get the current augmentation order and settings
        params = self.settings_tab.get_augmentation_params()
        params.update({
            'image_dir': os.path.join(self.dataset_root, 'images'),
            'label_dir': os.path.join(self.dataset_root, 'labels'),
            'augmented_image_dir': os.path.join(self.output_dir, 'images'),
            'augmented_label_dir': os.path.join(self.output_dir, 'labels'),
            'coco_image_folder': self.overlay_image_dir if self.overlay_image_dir else "",
        })

        # Create and configure worker
        self.worker = AugmentationWorker(params)
        self.worker.progress.connect(self.progress_dialog.progress_bar.setValue)
        self.worker.progress.connect(self.update_progress)  
        self.worker.progress_log.connect(lambda msg: self.progress_dialog.append_log(msg))
        self.worker.finished.connect(self.handle_completion)
        self.worker.error.connect(self.handle_augmentation_error)
        
        # Start time tracking for progress estimation
        self.progress_dialog.start_tracking()
        
        # Start processing
        self.worker.start()
        self.progress_dialog.exec()  # Show the dialog and wait for completion
    
    def update_progress(self, value):
        """Relay progress updates to the progress dialog"""
        self.progress_dialog.update_progress(
            value, 
            self.worker.processed_files if hasattr(self.worker, 'processed_files') else 0,
            self.worker.total_files if hasattr(self.worker, 'total_files') else 0
        )
        
    def handle_cancellation(self):
        """Handle cancellation from the progress dialog"""
        self.worker.cancel()
        self.progress_dialog.close()
    
    def handle_completion(self):
        """Handle successful completion of the augmentation process"""
        if not self.progress_dialog.is_cancelled:
            # Clear the image cache so that re-displayed images are freshly loaded
            self.image_cache.clear()

            self.image_viewer_tab.show_image()

            # Close the progress dialog
            self.progress_dialog.close()
            
            # Calculate total elapsed time
            elapsed_time = self.worker.end_time - self.worker.start_time
            formatted_time = self.progress_dialog.format_elapsed_time(elapsed_time)
            
            # Get number of processed and augmented files
            total_processed = self.worker.processed_files
            total_augmented = self.worker.augmented_files
            total_skipped = total_processed - total_augmented
            
            # Create a detailed message with statistics
            message = (
                f"Augmentation process completed successfully!\n\n"
                f"Total time: {formatted_time}\n"
                f"Files processed: {total_processed}\n"
                f"Files augmented: {total_augmented}\n"
                f"Files skipped: {total_skipped}\n"
            )
            
            QMessageBox.information(self, "Augmentation Complete", message)
            
    def handle_augmentation_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during augmentation: {error_msg}")