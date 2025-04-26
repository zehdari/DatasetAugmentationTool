from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt
import time

class AugmentationProgress(QWidget):
    """Component for showing augmentation progress"""
    
    def __init__(self, parent, progress_dialog):
        super().__init__()
        self.parent = parent
        self.progress_dialog = progress_dialog
        
        self.is_cancelled = False
        self.start_time = 0
        self.last_time_update = 0
        
        self.setup_progress_dialog()
        
    def setup_progress_dialog(self):
        """Set up the progress dialog for augmentation process"""
        layout = QVBoxLayout(self.progress_dialog)
        
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
        
        # Add our single cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.handle_cancellation)
        layout.addWidget(cancel_button)

        # Remove the default label from QProgressDialog if exists
        if self.progress_dialog.findChild(QLabel):
            self.progress_dialog.findChild(QLabel).hide()
            
    def start_tracking(self):
        """Start progress tracking"""
        self.is_cancelled = False
        self.start_time = time.time()
        self.last_time_update = self.start_time
        
    def update_progress(self, value):
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
            self.progress_bar.setValue(value)

            # Throttle ETA updates to once a second
            if current_time - self.last_time_update >= 1.0:
                done = self.parent.parent.worker.processed_files
                total = self.parent.parent.worker.total_files

                if done > 0 and total > done:
                    avg_per_file = elapsed / done
                    remaining = (total - done) * avg_per_file
                else:
                    remaining = 0.0

                self.time_label.setText(
                    f"Estimated remaining: {self.format_elapsed_time(remaining)}"
                )
                self.last_time_update = current_time

        except RuntimeError:
            # Widget deleted, ignore
            pass
            
    def log_message(self, message):
        """Add message to log text"""
        self.log_text.append(message)
            
    def handle_cancellation(self):
        """Handle user cancellation of the augmentation process"""
        self.is_cancelled = True
        self.parent.parent.worker.cancel()
        self.log_text.append("\nCancelling...")
        self.parent.parent.progress_dialog.close()

    def format_elapsed_time(self, seconds):
        """Format elapsed time into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:d}:{secs:02d}"