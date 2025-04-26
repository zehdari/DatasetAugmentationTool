from PyQt6.QtWidgets import (QProgressDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt
import time

class AugmentationProgressDialog(QProgressDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCancelButton(None)
        self.setWindowTitle("Processing Images")
        self.setMinimumWidth(600)
        self.setMinimumHeight(300)
        self.setAutoClose(True)
        self.setAutoReset(True)
        
        # Initialize time tracking variables
        self.start_time = 0
        self.last_time_update = 0
        self.is_cancelled = False
        
        # Set up UI components
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Top header with progress and time
        header_layout = QHBoxLayout()
        
        # Progress label on the left
        self.progress_label = QLabel("Starting...")
        header_layout.addWidget(self.progress_label, 1)  # Give it stretch factor
        
        # Elapsed time in the top right with label
        elapsed_layout = QHBoxLayout()
        elapsed_time_descriptor = QLabel("Elapsed Time:")
        self.elapsed_time_label = QLabel("0:00")
        self.elapsed_time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        elapsed_layout.addWidget(elapsed_time_descriptor)
        elapsed_layout.addWidget(self.elapsed_time_label)
        header_layout.addLayout(elapsed_layout)
        
        layout.addLayout(header_layout)
        
        # Time remaining estimate
        self.time_label = QLabel("Estimated remaining: Calculating...")
        layout.addWidget(self.time_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Add text area for logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        layout.addWidget(self.log_text)
        
        # Add cancel button
        self.cancel_button = QPushButton("Cancel")
        layout.addWidget(self.cancel_button)
        
        # Hide the default label from QProgressDialog
        self.findChild(QLabel).hide()
        
    def connect_cancel_button(self, cancel_handler):
        self.cancel_button.clicked.connect(cancel_handler)
        
    def start_tracking(self):
        """Initialize time tracking for the progress dialog"""
        self.start_time = time.time()
        self.last_time_update = self.start_time
        self.is_cancelled = False
        
    def update_progress(self, value, processed_files=None, total_files=None):
        """Update progress and time estimates using linear extrapolation."""
        # Skip if cancelled or no progress yet
        if self.is_cancelled or value <= 0:
            return

        try:
            current_time = time.time()
            elapsed = current_time - self.start_time

            # Update elapsed display
            self.elapsed_time_label.setText(self.format_elapsed_time(elapsed))

            # Update progress %
            self.progress_label.setText(f"Progress: {value}%")

            # Throttle ETA updates to once a second
            if current_time - self.last_time_update >= 1.0 and processed_files is not None and total_files is not None:
                if processed_files > 0 and total_files > processed_files:
                    avg_per_file = elapsed / processed_files
                    remaining = (total_files - processed_files) * avg_per_file
                else:
                    remaining = 0.0

                self.time_label.setText(
                    f"Estimated remaining: {self.format_elapsed_time(remaining)}"
                )
                self.last_time_update = current_time

        except RuntimeError:
            # Widget deleted, ignore
            pass
            
    def append_log(self, message):
        """Add a message to the log text area"""
        self.log_text.append(message)
        
    def format_elapsed_time(self, seconds):
        """Format elapsed time into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:d}:{secs:02d}"
            
    def handle_cancellation(self):
        """Mark the process as cancelled"""
        self.is_cancelled = True
        self.append_log("\nCancelling...")