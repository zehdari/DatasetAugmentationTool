import sys
import os
import re
import random
import time
import queue
from collections import Counter
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QSlider, QMessageBox, QGroupBox, QFormLayout,
                             QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QGridLayout, QSplitter,
                             QListWidget, QListWidgetItem, QSizePolicy, QScrollArea, QTabWidget, QColorDialog, 
                             QAbstractItemView, QProgressDialog, QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QObject, QEvent, QPointF, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QImage
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from augment_data import augment_image
import concurrent.futures
import gc
from dataclasses import dataclass
from typing import List, Dict, Any
import hashlib
import json
import yaml
import psutil
import threading

@dataclass
class BatchItem:
    root: str
    file: str
    current_subfolder: str
    current_augmented_image_dir: str
    current_augmented_label_dir: str
    params: Dict[str, Any]

def process_batch(batch: List[BatchItem]) -> List[tuple[bool, str]]:
    """Process a batch of images in parallel"""
    results = []
    for item in batch:
        try:
            # Setup paths
            img_path = os.path.join(item.root, item.file)
            lbl_path = os.path.join(item.params['label_dir'],
                                os.path.relpath(img_path, item.params['image_dir'])
                                ).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                                
            # Check if we should skip this image
            augmented_img_path = os.path.join(item.current_augmented_image_dir, item.file)
            augmented_lbl_path = os.path.join(item.current_augmented_label_dir,
                                            item.file.replace('.jpg', '.txt')
                                            .replace('.jpeg', '.txt')
                                            .replace('.png', '.txt'))
                                            
            if item.params.get('skip_existing', True) and os.path.exists(augmented_img_path) and os.path.exists(augmented_lbl_path):
                results.append((True, f"Skipped existing: {item.file}"))
                continue

            # Read and process image
            image = cv2.imread(img_path)
            if image is None:
                results.append((False, f"Could not read image: {img_path}"))
                continue
                
            (h, w) = image.shape[:2]

            # Process labels
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    polygons = []
                    class_ids = []
                    for line in lines:
                        parts = line.strip().split()
                        class_id = parts[0]
                        class_ids.append(class_id)
                        polygon = [(float(parts[i]), float(parts[i + 1])) 
                                for i in range(1, len(parts), 2)]
                        polygons.append(polygon)
            else:
                results.append((False, f"Label file not found: {lbl_path}"))
                continue

            # Extract augmentation parameters
            augmentation_params = {
                'skip_augmentations': item.params['skip_augmentations'],
                'mirror_weights': item.params['mirror_weights'],
                'crop_weights': item.params['crop_weights'],
                'overlay_weights': item.params['overlay_weights'],
                'rotate_weights': item.params['rotate_weights'],
                'rotation_random_vs_90_weights': item.params['rotation_random_vs_90_weights'],
                'overlay_min_max_scale': item.params['overlay_min_max_scale'],
                'maintain_aspect_ratio_weights': item.params['maintain_aspect_ratio_weights'],
                'zoom_weights': item.params['zoom_weights'],
                'zoom_in_vs_out_weights': item.params['zoom_in_vs_out_weights'],
                'zoom_padding': item.params['zoom_padding'],
                'coco_image_folder': item.params['coco_image_folder']
            }

            # Perform augmentation
            from augment_data import augment_image
            augmented_image, augmented_polygons = augment_image(
                image=image,
                polygons=polygons,
                current_subfolder=item.current_subfolder,
                class_ids=class_ids,
                h=h,
                w=w,
                **augmentation_params
            )

            # Save results
            os.makedirs(os.path.dirname(augmented_img_path), exist_ok=True)
            cv2.imwrite(augmented_img_path, augmented_image)
            
            os.makedirs(os.path.dirname(augmented_lbl_path), exist_ok=True)
            with open(augmented_lbl_path, 'w') as f:
                for polygon in augmented_polygons:
                    line = ' '.join(map(str, [polygon[0]] + 
                                    [coord for point in polygon[1:] for coord in point]))
                    f.write(line + '\n')
            
            results.append((True, f"Successfully processed: {item.file}"))
            
        except Exception as e:
            results.append((False, f"Error processing {item.file}: {str(e)}"))
            
    return results

def process_single_image_worker(args):
    """Standalone function for processing a single image."""
    try:
        relative_path, label_path, subfolder, params, overlay_image = args
        
        # Set consistent seed for this file
        seed = int(hashlib.md5(relative_path.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        
        # Construct full paths
        img_path = os.path.join(params['image_dir'], relative_path)
        
        # Construct output paths maintaining folder structure
        augmented_img_path = os.path.join(params['augmented_image_dir'], relative_path)
        augmented_lbl_path = os.path.join(
            params['augmented_label_dir'],
            os.path.splitext(relative_path)[0] + '.txt'
        )
        
        # Create output directories
        os.makedirs(os.path.dirname(augmented_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(augmented_lbl_path), exist_ok=True)
        
        # Check if we should skip this image
        if (params.get('skip_existing', True) and 
            os.path.exists(augmented_img_path) and 
            os.path.exists(augmented_lbl_path)):
            return True, f"Skipped existing: {relative_path}"
        
        # Read and validate image
        image = cv2.imread(img_path)
        if image is None:
            return False, f"Could not read image: {img_path}"
        
        h, w = image.shape[:2]
        
        # Read and parse labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            polygons = []
            class_ids = []
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]
                class_ids.append(class_id)
                coords = [(float(parts[i]), float(parts[i + 1])) 
                         for i in range(1, len(parts), 2)]
                polygons.append(coords)
        
        # Extract the subset of parameters needed for augmentation
        augmentation_params = {
            'skip_augmentations': params['skip_augmentations'],
            'mirror_weights': params['mirror_weights'],
            'crop_weights': params['crop_weights'],
            'overlay_weights': params['overlay_weights'],
            'rotate_weights': params['rotate_weights'],
            'rotation_random_vs_90_weights': params['rotation_random_vs_90_weights'],
            'overlay_min_max_scale': params['overlay_min_max_scale'],
            'maintain_aspect_ratio_weights': params['maintain_aspect_ratio_weights'],
            'zoom_weights': params['zoom_weights'],
            'zoom_in_vs_out_weights': params['zoom_in_vs_out_weights'],
            'zoom_padding': params['zoom_padding'],
            'augmentation_order': params.get('augmentation_order')
        }
        
        # Perform augmentation
        from augment_data import augment_image
        augmented_image, augmented_polygons = augment_image(
            image=image,
            polygons=polygons,
            current_subfolder=subfolder,
            class_ids=class_ids,
            h=h,
            w=w,
            coco_image=overlay_image,  # Pass the overlay image directly
            **augmentation_params  # Pass all parameters at once
        )
        
        # Save augmented image
        cv2.imwrite(augmented_img_path, augmented_image)
        
        # Save augmented labels
        with open(augmented_lbl_path, 'w') as f:
            for polygon in augmented_polygons:
                line = ' '.join(map(str, [polygon[0]] + 
                               [coord for point in polygon[1:] for coord in point]))
                f.write(line + '\n')
        
        return True, f"Successfully processed: {relative_path}"
        
    except Exception as e:
        return False, f"Error processing {relative_path}: {str(e)}"

class OverlayProvider:
    """
    Keeps up to `max_buffers` overlay images pre-loaded in a thread-safe queue.
    Images are cycled without repeats until the entire pool has been used.
    """

    def __init__(self, image_folder: str, max_buffers: int = 10):
        self.image_folder = image_folder
        # list of all files in the folder
        self.files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                      and os.path.isfile(os.path.join(image_folder, f))]
        if not self.files:
            raise RuntimeError(f"No images found in {image_folder}")

        # a shuffled pool of indices, so we never repeat until exhausted
        self._lock = threading.Lock()
        self.idx_pool = list(range(len(self.files)))
        random.shuffle(self.idx_pool)

        # a bounded queue that holds CV2 images
        self.q = queue.Queue(maxsize=max_buffers)
        self._shutdown = threading.Event()

        # start the background loader
        self.loader = threading.Thread(target=self._loader, daemon=True)
        self.loader.start()

    def _loader(self):
        while not self._shutdown.is_set():
            try:
                # block until there’s space in the queue
                img = self._load_one()
                self.q.put(img, timeout=0.1)
            except queue.Full:
                # queue is full, retry until shutdown
                continue

    def _load_one(self):
        # refill & reshuffle when we’ve used all
        with self._lock:
            if not self.idx_pool:
                self.idx_pool = list(range(len(self.files)))
                random.shuffle(self.idx_pool)
            idx = self.idx_pool.pop()

        path = os.path.join(self.image_folder, self.files[idx])
        img = cv2.imread(path)
        if img is None:
            # if a file failed to read, skip it
            return self._load_one()
        return img

    def get_overlay(self, timeout=None):
        """
        Retrieve the next pre-loaded image (blocks until one’s available).
        Returns None only if `timeout` is given and expires.
        """
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        """Shut down the loader thread and drain the queue."""
        self._shutdown.set()
        self.loader.join()
        # clear out any images
        while not self.q.empty():
            self.q.get_nowait()

class AugmentationWorker(QThread):
    progress = pyqtSignal(int)
    progress_log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_cancelled = False
        self.total_files = 0
        self.processed_files = 0
        # New attributes for tracking stats
        self.augmented_files = 0  # Track files actually augmented (not skipped)
        self.start_time = None
        self.end_time = None
        self.semaphore = None
        
        # Calculate optimal batch sizes based on system memory
        self.calculate_memory_adaptive_parameters()
        
        # Initialize asynchronous image loader if overlay directory is provided
        if self.params.get('coco_image_folder'):
            self.progress_log.emit(f"Initializing overlay queue with 3 buffers")
            self.image_loader = OverlayProvider(self.params['coco_image_folder'], max_buffers=3)
        else:
            self.image_loader = None
    
    def calculate_memory_adaptive_parameters(self):
        """Calculate batch size and other parameters based on available system memory."""
        try:
            # Get memory information
            mem = psutil.virtual_memory()
            total_memory = mem.total
            available_memory = mem.available
            
            # Memory allocation for overlay images (set to 20% of available memory)
            overlay_memory_allocation = available_memory * 0.2
            
            # Estimate average image size if overlay directory is provided
            avg_image_size = self.estimate_average_image_size()
            
            if avg_image_size > 0:
                # Calculate batch size based on available memory and average image size
                # Add 20% overhead for other operations
                max_images = int(overlay_memory_allocation / (avg_image_size * 1.2))
                
                # We need two buffers, so divide by 2 and add some margin
                max_images = int(max_images / 2 * 0.9)
                
                # Clamp batch size between reasonable values
                self.batch_size = max(20, min(300, max_images))
                
                self.progress_log.emit(f"Memory-adaptive parameters: batch_size={self.batch_size}")
            else:
                # Default values if calculation fails
                self.batch_size = 100
                self.progress_log.emit("Using default batch parameters: batch_size=100")
        except Exception as e:
            # Fall back to default values on error
            self.batch_size = 100
            self.progress_log.emit(f"Error calculating memory parameters: {str(e)}")
            self.progress_log.emit("Using default batch parameters: batch_size=100")
    
    def estimate_average_image_size(self):
        """Estimate the average size of overlay images in bytes."""
        try:
            if not self.params.get('coco_image_folder') or not os.path.exists(self.params['coco_image_folder']):
                return 0
                
            image_folder = self.params['coco_image_folder']
            image_files = [f for f in os.listdir(image_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                          and os.path.isfile(os.path.join(image_folder, f))]
            
            if not image_files:
                return 0
                
            # Sample up to 10 random images to estimate average size
            sample_size = min(10, len(image_files))
            sample_files = random.sample(image_files, sample_size)
            
            # Calculate file sizes
            sizes = []
            for file in sample_files:
                file_path = os.path.join(image_folder, file)
                # Get file size on disk
                file_size = os.path.getsize(file_path)
                
                # Load image to estimate in-memory size
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        # Calculate in-memory size (height × width × channels × bytes per channel)
                        mem_size = img.nbytes
                        # Use the larger of file size or memory size
                        sizes.append(max(file_size, mem_size))
                except Exception:
                    # If loading fails, just use file size
                    sizes.append(file_size)
            
            # Calculate average size with outlier protection
            if sizes:
                if len(sizes) > 3:
                    # Remove outliers for more accurate estimation
                    sizes.sort()
                    sizes = sizes[1:-1]  # Remove smallest and largest
                    
                avg_size = sum(sizes) / len(sizes)
                self.progress_log.emit(f"Average image size: {avg_size/1024:.1f} KB")
                return avg_size
            return 0
        except Exception as e:
            self.progress_log.emit(f"Error estimating image size: {str(e)}")
            return 0

    def preload_overlay_images(self):
        """Preload overlay images from the specified directory."""
        try:
            image_folder = self.params['coco_image_folder']
            self.progress_log.emit(f"Preloading overlay images from {image_folder}...")
            
            # Get list of all image files
            self.all_image_files = [f for f in os.listdir(image_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                        and os.path.isfile(os.path.join(image_folder, f))]
            
            # Initialize tracking variables
            self.used_image_indices = set()
            
            # Load initial batch of images
            self.load_next_batch_of_images()
            
            self.progress_log.emit(f"Successfully preloaded {len(self.overlay_images)} overlay images")
        except Exception as e:
            self.error.emit(f"Error preloading overlay images: {str(e)}")
            self.overlay_images = []

    def load_next_batch_of_images(self):
        """Load a new batch of overlay images."""
        try:
            image_folder = self.params['coco_image_folder']
            
            # Clear current images to free memory
            self.overlay_images = []
            
            # Reset the usage counter when loading a new batch
            self.image_usage_counter = 0
            
            # Figure out which images haven't been used
            available_indices = set(range(len(self.all_image_files))) - self.used_image_indices
            
            # If all images have been used, reset the tracking
            if not available_indices or len(available_indices) < self.batch_size:
                self.progress_log.emit("All overlay images used, refreshing selection...")
                self.used_image_indices = set()
                available_indices = set(range(len(self.all_image_files)))
            
            # Select batch_size random indices from available images
            batch_size = min(self.batch_size, len(available_indices))
            batch_indices = random.sample(list(available_indices), batch_size)
            
            # Mark these indices as used
            self.used_image_indices.update(batch_indices)
            
            # Load the images
            for idx in batch_indices:
                filename = self.all_image_files[idx]
                try:
                    img_path = os.path.join(image_folder, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.overlay_images.append(img)
                except Exception as e:
                    self.progress_log.emit(f"Error loading overlay image {filename}: {str(e)}")
            
            self.progress_log.emit(f"Loaded {len(self.overlay_images)} new overlay images")
            
            # Force garbage collection after loading new batch
            gc.collect()
            
        except Exception as e:
            self.error.emit(f"Error loading batch of overlay images: {str(e)}")

    def get_random_overlay_image(self):
        """Get a random overlay image using the asynchronous loader."""
        if self.image_loader:
            return self.image_loader.get_overlay()
        return None
    
    def calculate_optimal_workers(self):
        """Calculate the optimal number of worker processes based on system resources."""
        try:
            # Get system memory information
            mem = psutil.virtual_memory()
            available_memory = mem.available
            
            # Sample the first image to estimate memory usage
            image_label_pairs = self.collect_image_label_pairs()
            if not image_label_pairs:
                return os.cpu_count() or 1
                
            img_path = os.path.join(self.params['image_dir'], image_label_pairs[0][0])
            img_size = os.path.getsize(img_path)
            label_size = os.path.getsize(image_label_pairs[0][1])
            
            # Estimate memory needed per process (4x for safety margin)
            memory_per_process = (img_size + label_size) * 4
            
            # Calculate max workers based on available memory (use 60% of available memory)
            # Reserve some memory for overlay images and UI
            memory_based_workers = int((available_memory * 0.6) // memory_per_process)
            
            # Get CPU count but limit to physical cores for better performance
            try:
                physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
            except:
                physical_cores = os.cpu_count() or 1
            
            # Use the minimum of memory-based or CPU-based worker count
            optimal_workers = max(1, min(memory_based_workers, physical_cores))
            
            self.progress_log.emit(f"Memory-based worker limit: {memory_based_workers}, CPU-based limit: {physical_cores}")
            return optimal_workers
            
        except Exception as e:
            self.progress_log.emit(f"Error calculating workers: {str(e)}")
            return max(1, (os.cpu_count() or 2) // 2)  # Default to half of available logical cores
    
    def calculate_processing_batch_size(self):
        """Calculate the optimal batch size for task processing."""
        try:
            # Get available memory (in GB)
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            
            # Base batch size on available memory
            if available_memory > 16:  # More than 16GB available
                batch_size = 100
            elif available_memory > 8:  # 8-16GB available
                batch_size = 50
            elif available_memory > 4:  # 4-8GB available
                batch_size = 25
            else:  # Less than 4GB available
                batch_size = 10
                
            # Cap batch size to 20% of total files for small datasets
            max_batch = max(10, int(self.total_files * 0.2))
            return min(batch_size, max_batch)
            
        except Exception:
            # Default to reasonable batch size if calculation fails
            return min(50, self.total_files)
    
    def run(self):
        """Execute the augmentation process with optimized parallel processing."""
        try:
            # Record start time
            self.start_time = time.time()
            
            # Collect image-label pairs
            image_label_pairs = self.collect_image_label_pairs()
            
            if not image_label_pairs:
                self.progress_log.emit("No image files found to process.")
                self.finished.emit()
                return
            
            self.total_files = len(image_label_pairs)
            self.progress_log.emit(f"Total files to process: {self.total_files}")
            
            # Calculate optimal number of workers
            num_workers = self.calculate_optimal_workers()
            self.progress_log.emit(f"Using {num_workers} worker processes")
            
            # Initialize processing
            self.processed_files = 0
            
            # Calculate optimal processing batch size
            processing_batch_size = self.calculate_processing_batch_size()
            self.progress_log.emit(f"Processing in batches of {processing_batch_size} files")
            
            # Process images in parallel with proper resource management
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit tasks in smaller batches to control memory usage
                remaining_tasks = list(enumerate(image_label_pairs))
                
                while remaining_tasks and not self.is_cancelled:
                    current_batch = remaining_tasks[:processing_batch_size]
                    remaining_tasks = remaining_tasks[processing_batch_size:]
                    
                    futures = []
                    for idx, (relative_path, label_path, subfolder) in current_batch:
                        if self.is_cancelled:
                            break
                        
                        # Create a copy of the parameters
                        task_params = self.params.copy()
                        
                        # Get a random overlay image from async loader
                        overlay_img = self.get_random_overlay_image()
                        
                        # Submit the task with a unique identifier
                        future = executor.submit(
                            process_single_image_worker,
                            (relative_path, label_path, subfolder, task_params, overlay_img)
                        )
                        # Store task index with future for tracking
                        futures.append((idx, future))
                    
                    # Create a mapping of futures to their indices for tracking
                    future_to_idx = {f: i for i, f in futures}
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed([f for _, f in futures]):
                        if self.is_cancelled:
                            break
                        
                        try:
                            success, message = future.result()
                            
                            # Update progress indicators - these run in main thread context
                            self.processed_files += 1
                            progress = int(self.processed_files * 100 / self.total_files)
                            self.progress.emit(progress)
                            
                            if success:
                                # If the message doesn't contain "Skipped existing", increment the augmented files counter
                                if "Skipped existing" not in message:
                                    self.augmented_files += 1
                                self.progress_log.emit(f"[{self.processed_files}/{self.total_files}] {message}")
                            else:
                                self.progress_log.emit(f"Error: {message}")
                            
                            # Force UI update by yielding to event loop for a moment
                            QApplication.processEvents()
                                
                        except Exception as e:
                            self.error.emit(f"Task error: {str(e)}")
                            continue
                    
                    # Force garbage collection after each batch
                    gc.collect()
            
            if not self.is_cancelled:
                self.end_time = time.time()
                self.progress_log.emit("\nProcessing complete!")
                self.finished.emit()
                
        except Exception as e:
            self.end_time = time.time()
            self.error.emit(str(e))
        finally:
            # If end_time wasn't set for some reason, set it now
            if self.end_time is None:
                self.end_time = time.time()
            # Clean up resources
            if self.image_loader:
                self.image_loader.close()
            gc.collect()

    def atoi(self, text):
        """Helper function for natural sort order"""
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        """Sort text strings in natural order"""
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def collect_image_label_pairs(self):
        """Collect matching image and label file pairs with proper relative paths."""
        image_label_pairs = []
        for root, _, files in os.walk(self.params['image_dir']):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Get relative path from image directory
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(image_path, self.params['image_dir'])
                    subfolder = os.path.dirname(relative_path)
                    
                    # Construct label path maintaining folder structure
                    label_path = os.path.join(
                        self.params['label_dir'],
                        os.path.splitext(relative_path)[0] + '.txt'
                    )
                    
                    if os.path.exists(label_path):
                        image_label_pairs.append((relative_path, label_path, subfolder))
        
        return sorted(image_label_pairs, key=lambda x: self.natural_keys(x[0]))

    def cancel(self):
        """Cancel the augmentation process."""
        self.is_cancelled = True
        self.progress_log.emit("\nCancelling...")
        
        # Shut down image loader if active
        if hasattr(self, 'image_loader') and self.image_loader:
            self.image_loader.close()
    
class ImageCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        
    def get_image(self, path):
        if path in self.cache:
            return self.cache[path].copy()
        
        image = cv2.imread(path)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = image
        return image.copy()

    def clear(self):
        self.cache.clear()

class ClickFilter(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            for widget in obj.findChildren(QLineEdit):
                widget.clearFocus()
        return super().eventFilter(obj, event)

class CustomLineEdit(QLineEdit):
    def focusOutEvent(self, event):
        self.deselect()
        super().focusOutEvent(event)

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

class AugmentDatasetGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.image_cache = ImageCache(max_size=100)
        self.config_manager = ConfigManager('augmentation_config.json')

        self.dataset_root = ""
        self.overlay_image_dir = ""
        self.output_dir = ""
        self.skip_augmentations = {
            'Zoom': [],
            'Crop': [],
            'Rotate': [],
            'Mirror': [],
            'Overlay': []
        }
        self.image_paths = []  # List to store image paths
        self.folder_images = []
        self.label_paths = {}  # Dictionary to store label paths
        self.current_image_index = 0  # To store the index of the current displayed image
        self.current_image_path = ""
        self.folder_name = ""
        self.class_colors = {}  # Dictionary to store class colors
        self.augmented_image = None
        self.augmented_polygons = None
        self.augmented_image_original_dims = None

        self.rotation_random_vs_90 = [25, 75]
        self.zoom_in_vs_out_weights = [40, 60]
        self.zoom_in_min_padding = 0.05
        self.zoom_in_max_padding = 0.5
        self.zoom_out_min_padding = 0.1
        self.zoom_out_max_padding = 0.8
        self.zoom_padding = [self.zoom_in_min_padding, self.zoom_in_max_padding, self.zoom_out_min_padding, self.zoom_out_max_padding]
        self.maintain_aspect_ratio_weights = [50, 50]
        self.overlay_min_max_scale = [0.3, 1.0]
        self.id_to_label = {}
        self.label_to_id = {}

        self.show_labels = True
        self.show_polygons = True 
        self.show_bounding_boxes = False 
        self.show_points = False 

        self.output_dir_set = False  # Flag to track if output directory has been set

        self.show_original = False
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dataset Augmentation GUI')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        # Tabs
        tab_widget = QTabWidget()
        self.augmentation_settings_tab = QWidget()
        self.image_viewer_tab = QWidget()
        self.dataset_stats_tab = QWidget()

        tab_widget.addTab(self.augmentation_settings_tab, "Settings")
        tab_widget.addTab(self.image_viewer_tab, "Image Viewer")
        tab_widget.addTab(self.dataset_stats_tab, "Dataset Stats")

        self.init_augmentation_settings_tab()
        self.init_image_viewer_tab()
        self.init_dataset_stats_tab()

        main_layout.addWidget(tab_widget)

        # Bottom button layout
        bottom_button_layout = QHBoxLayout()  # Use vertical layout to stack elements vertically

        # Horizontal layout for Save and Load buttons

        # Run button on its own row
        self.run_btn = QPushButton("Run Augmentation")
        self.run_btn.clicked.connect(self.run_augmentation)

        bottom_button_layout.addWidget(self.run_btn)          # Second row with Run button

        # Add bottom button layout to main layout
        main_layout.addLayout(bottom_button_layout)


        self.setLayout(main_layout)
        self.installEventFilter(ClickFilter(self))

    def save_current_config(self):
        # Get the current order of sliders
        slider_order = []
        for i in range(self.sliders_list.count()):
            item_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
            # Find the slider widget in the layout
            for j in range(item_widget.layout().count()):
                widget = item_widget.layout().itemAt(j).widget()
                if isinstance(widget, QSlider):
                    # Find which attribute this slider corresponds to
                    for attr_name, attr_value in vars(self).items():
                        if attr_value is widget and attr_name in self.slider_to_augmentation_type:
                            slider_order.append(self.slider_to_augmentation_type[attr_name])
                            break
                    break
        
        config_data = {
            "augmentation_order": slider_order,
            "crop_probability": self.crop_slider.value(),
            "maintain_aspect_ratio": self.maintain_aspect_ratio_slider.value(),
            "mirror_probability": self.mirror_slider.value(),
            "overlay_probability": self.overlay_slider.value(),
            "rotate_probability": self.rotate_slider.value(),
            "rotation_random_vs_90": self.rotation_random_vs_90_slider.value(),
            "zoom_in_vs_out": self.zoom_in_vs_out_slider.value(),
            "zoom_probability": self.zoom_slider.value(),
            "skip_existing": self.skip_existing_checkbox.isChecked()
        }
        self.config_manager.save_config(config_data)

    def load_existing_config(self):
        config_data = self.config_manager.load_config()
        if config_data:
            # Set slider values
            self.crop_slider.setValue(config_data.get("crop_probability", 0))
            self.maintain_aspect_ratio_slider.setValue(config_data.get("maintain_aspect_ratio", 0))
            self.mirror_slider.setValue(config_data.get("mirror_probability", 0))
            self.overlay_slider.setValue(config_data.get("overlay_probability", 0))
            self.rotate_slider.setValue(config_data.get("rotate_probability", 0))
            self.rotation_random_vs_90_slider.setValue(config_data.get("rotation_random_vs_90", 0))
            self.zoom_in_vs_out_slider.setValue(config_data.get("zoom_in_vs_out", 0))
            self.zoom_slider.setValue(config_data.get("zoom_probability", 0))
            self.skip_existing_checkbox.setChecked(config_data.get("skip_existing", False))
            
            # Apply saved order if available
            if "augmentation_order" in config_data:
                self.reorder_sliders_from_config(config_data["augmentation_order"])
            
            self.update_sliders_state()
                
    def reorder_sliders_from_config(self, order):
        """Reorder sliders based on the order saved in the config"""
        # Get current information about all sliders
        slider_info = []
        for i in range(self.sliders_list.count()):
            item = self.sliders_list.item(i)
            widget = self.sliders_list.itemWidget(item)
            
            # Find which slider this corresponds to
            for j in range(widget.layout().count()):
                child_widget = widget.layout().itemAt(j).widget()
                if isinstance(child_widget, QSlider):
                    # Find which attribute this slider corresponds to
                    for attr_name, attr_value in vars(self).items():
                        if attr_value is child_widget and attr_name in self.slider_to_augmentation_type:
                            aug_type = self.slider_to_augmentation_type[attr_name]
                            # Store current value
                            value = child_widget.value()
                            slider_info.append({
                                'aug_type': aug_type,
                                'slider_attr': attr_name,
                                'value': value,
                                'index': i
                            })
                            break
                    break
        
        # Create a mapping from augmentation types to their info
        aug_type_to_info = {info['aug_type']: info for info in slider_info}
        
        # Temporarily remove all sliders
        for i in range(self.sliders_list.count() - 1, -1, -1):
            self.sliders_list.takeItem(i)
        
        # Re-add sliders in the desired order
        for aug_type in order:
            if aug_type in aug_type_to_info:
                info = aug_type_to_info[aug_type]
                
                # Find the corresponding slider_info entry
                for slider_item in self.slider_data:
                    if slider_item['object'] == info['slider_attr']:
                        # Re-add the slider with proper values
                        self.add_slider_to_list(
                            slider_item["name"], 
                            slider_item["object"], 
                            slider_item["value_object"]
                        )
                        
                        # Set the value to match the original
                        slider = getattr(self, slider_item["object"])
                        slider.setValue(info['value'])
                        break

    def init_augmentation_settings_tab(self):
        layout = QVBoxLayout()

        # Combined directory selection
        dir_group = QGroupBox("Select Directories")
        dir_layout = QFormLayout()
        self.dataset_label = QLabel("Not selected")
        self.overlay_label = QLabel("Not selected")
        self.output_dir_label = QLabel("Not selected")
        self.dataset_btn = QPushButton("Select Dataset Root")
        self.overlay_btn = QPushButton("Select Overlay Image Directory")
        self.output_dir_btn = QPushButton("Select Output Directory")
        self.dataset_btn.clicked.connect(self.select_dataset_root)
        self.overlay_btn.clicked.connect(self.select_overlay_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        dir_layout.addRow(self.dataset_btn, self.dataset_label)
        dir_layout.addRow(self.overlay_btn, self.overlay_label)
        dir_layout.addRow(self.output_dir_btn, self.output_dir_label)
        dir_group.setLayout(dir_layout)
        dir_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        layout.addWidget(dir_group)

        # Sliders and Skip Augmentations
        weights_skip_layout = QSplitter(Qt.Orientation.Horizontal)
        
        # Weights sliders with scroll area inside a group box
        weights_group = QGroupBox("Augmentation Settings")
        weights_layout = QVBoxLayout()  # Changed to QVBoxLayout to simplify

        self.sliders_list = QListWidget()
        self.sliders_list.setDragEnabled(True)
        self.sliders_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.sliders_list.setMinimumHeight(300) 
        
        self.sliders_list.setStyleSheet("""
            QListWidget::item:selected { 
                background: transparent; 
                color: black;
            }
            QListWidget::item:hover { 
                background: transparent; 
                border: none;
            }
            QListWidget::item:selected:active {
                background: transparent;
                color: black;
            }
            QListWidget::item:selected:!active {
                background: transparent;
                color: black;
            }
        """)
            
        self.slider_data = [
            {"name": "Mirror % Probability:", "object": "mirror_slider", "value_object": "mirror_value"},
            {"name": "Rotate % Probability:", "object": "rotate_slider", "value_object": "rotate_value"},
            {"name": "Rotation (0 to 360) vs 90 %: ", "object": "rotation_random_vs_90_slider", "value_object": "rotation_random_vs_90_value"},
            {"name": "Crop % Probability:", "object": "crop_slider", "value_object": "crop_value"},
            {"name": "Maintain Aspect Ratio on Crop %: ", "object": "maintain_aspect_ratio_slider", "value_object": "maintain_aspect_ratio_value"},
            {"name": "Zoom % Probability:", "object": "zoom_slider", "value_object": "zoom_value"},
            {"name": "Zoom In vs Out %: ", "object": "zoom_in_vs_out_slider", "value_object": "zoom_in_vs_out_value"},
            {"name": "Overlay % Probability:", "object": "overlay_slider", "value_object": "overlay_value"}
        ]

        self.slider_to_augmentation_type = {
            "mirror_slider": "mirror",
            "rotate_slider": "rotate",
            "rotation_random_vs_90_slider": "rotation_random_vs_90",
            "crop_slider": "crop",
            "maintain_aspect_ratio_slider": "maintain_aspect_ratio",
            "zoom_slider": "zoom",
            "zoom_in_vs_out_slider": "zoom_in_vs_out",
            "overlay_slider": "overlay"
        }
        for slider_info in self.slider_data:
            self.add_slider_to_list(slider_info["name"], slider_info["object"], slider_info["value_object"])
        
        weights_layout.addWidget(self.sliders_list)
        
        # Settings buttons group (ONLY ONE DEFINITION)
        settings_buttons_layout = QHBoxLayout()
        self.skip_existing_checkbox = QCheckBox("Skip Already Augmented Images")
        self.skip_existing_checkbox.setChecked(True)
        settings_buttons_layout.addWidget(self.skip_existing_checkbox)
        
        self.load_config_button = QPushButton("Load Config")
        self.save_config_button = QPushButton("Save Config")
        
        self.load_config_button.clicked.connect(self.load_existing_config)
        self.save_config_button.clicked.connect(self.save_current_config)
        
        settings_buttons_layout.addWidget(self.load_config_button)
        settings_buttons_layout.addWidget(self.save_config_button)
        
        # Add the settings buttons layout to the weights layout
        weights_layout.addLayout(settings_buttons_layout)
        weights_group.setLayout(weights_layout)

        # Skip Augmentations
        skip_colors_layout = QSplitter(Qt.Orientation.Vertical)

        self.skip_group = QGroupBox("Skip Augmentations for Folders")
        self.skip_layout = QVBoxLayout()
        self.skip_table = QTableWidget()
        self.skip_table.setColumnCount(7)  # Update the column count
        self.skip_table.setHorizontalHeaderLabels(['Folder', 'Zoom', 'Crop', 'Rotate', 'Mirror', 'Overlay', 'Skip All'])
        self.skip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.skip_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for col in range(1, 7):
            self.skip_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.skip_table.setColumnWidth(col, 50)
        self.skip_layout.addWidget(self.skip_table)
        self.skip_group.setLayout(self.skip_layout)
        
        skip_colors_layout.addWidget(self.skip_group)

        self.class_color_group = QGroupBox("Class Colors")
        self.class_colors_layout = QVBoxLayout()
        self.class_colors_table = QTableWidget()
        self.class_colors_table.setColumnCount(2)
        self.class_colors_table.setHorizontalHeaderLabels(['Class', 'Color'])
        self.class_colors_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_colors_table.itemClicked.connect(self.on_color_cell_clicked)
        self.class_colors_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.class_colors_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.class_colors_layout.addWidget(self.class_colors_table)
        self.class_color_group.setLayout(self.class_colors_layout)
        
        skip_colors_layout.addWidget(self.class_color_group)
        skip_colors_layout.setCollapsible(0, False)
        skip_colors_layout.setCollapsible(1, False)

        weights_skip_layout.addWidget(weights_group)
        weights_skip_layout.addWidget(skip_colors_layout)
        weights_skip_layout.setSizes([800, 400])  # Initial sizes of the panels
        weights_skip_layout.setCollapsible(0, False)
        weights_skip_layout.setCollapsible(1, False)

        # Set minimum sizes
        self.skip_group.setMinimumWidth(400)
        weights_skip_layout.setMinimumWidth(1200)
        weights_skip_layout.setHandleWidth(10)

        layout.addWidget(weights_skip_layout)
        self.augmentation_settings_tab.setLayout(layout)

    def add_slider_to_list(self, name, slider_attr, value_attr):
        # Create a widget to hold the slider row
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create drag handle label
        drag_handle = QLabel("≡")  # Using equal sign to create a simple handle icon
        drag_handle.setFixedWidth(20)
        drag_handle.setStyleSheet("""
            font-size: 18px; 
            color: #999;
            padding: 2px;
            border-radius: 3px;
        """)
        
        # Create label
        label = QLabel(name)
        label.setMinimumWidth(180)  # Reduced width slightly to make room for handle
        
        # Create slider and value
        slider, value_edit = self.create_slider()
        
        # Store references to these widgets
        setattr(self, slider_attr, slider)
        setattr(self, value_attr, value_edit)
        
        # If we already have values for this slider, use them
        if hasattr(self, slider_attr) and isinstance(getattr(self, slider_attr), QSlider):
            old_slider = getattr(self, slider_attr)
            if old_slider and hasattr(old_slider, 'value'):
                # Try to get the value of the previous slider
                try:
                    slider.setValue(old_slider.value())
                    value_edit.setText(str(old_slider.value()))
                except Exception:
                    # If it fails, just use default values
                    pass
        
        # Add to layout
        item_layout.addWidget(drag_handle)
        item_layout.addWidget(label)
        item_layout.addWidget(slider, 1)  # Give slider stretch factor
        item_layout.addWidget(value_edit)
        
        # Add row to list widget
        list_item = QListWidgetItem(self.sliders_list)
        list_item.setSizeHint(item_widget.sizeHint())
        self.sliders_list.addItem(list_item)
        self.sliders_list.setItemWidget(list_item, item_widget)

    def init_image_viewer_tab(self):
        layout = QVBoxLayout()

        self.image_viewer_layout = QVBoxLayout()
        folder_list_layout = QHBoxLayout()
        folder_and_button_layout = QVBoxLayout()

        self.folder_list = QListWidget()
        self.folder_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.folder_list.setMaximumHeight(125)
        self.folder_list.setMinimumHeight(50)
        self.folder_list.itemClicked.connect(self.display_images)
        folder_and_button_layout.addWidget(self.folder_list)

        self.show_original_btn = QPushButton("Show Original Image")
        self.show_original_btn.clicked.connect(self.show_original_image)
        folder_and_button_layout.addWidget(self.show_original_btn)

        folder_list_layout.addLayout(folder_and_button_layout)

        checkboxes_button_layout = QVBoxLayout()
        self.labels_checkbox = QCheckBox("Show Labels")
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.stateChanged.connect(self.toggle_labels)
        checkboxes_button_layout.addWidget(self.labels_checkbox)

        self.polygons_checkbox = QCheckBox("Show Polygons")
        self.polygons_checkbox.setChecked(True)
        self.polygons_checkbox.stateChanged.connect(self.toggle_polygons)
        checkboxes_button_layout.addWidget(self.polygons_checkbox)

        self.bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(False)
        self.bbox_checkbox.stateChanged.connect(self.toggle_bounding_boxes)
        checkboxes_button_layout.addWidget(self.bbox_checkbox)

        self.points_checkbox = QCheckBox("Show Points")
        self.points_checkbox.setChecked(False)
        self.points_checkbox.stateChanged.connect(self.toggle_points)
        checkboxes_button_layout.addWidget(self.points_checkbox)

        self.augment_single_btn = QPushButton("Preview Augmentation")
        self.augment_single_btn.clicked.connect(self.augment_current_image)
        checkboxes_button_layout.addWidget(self.augment_single_btn)

        self.save_preview_btn = QPushButton("Save Current Preview")
        self.save_preview_btn.clicked.connect(self.save_current_preview)
        checkboxes_button_layout.addWidget(self.save_preview_btn)

        folder_list_layout.addLayout(checkboxes_button_layout)

        folder_list_layout.setStretch(0, 1)
        folder_list_layout.setStretch(1, 0)

        self.image_viewer_layout.addLayout(folder_list_layout)

        self.image_name_label = QLabel("")
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_viewer_layout.addWidget(self.image_name_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumHeight(150)
        self.image_viewer_layout.addWidget(self.image_label)

        self.image_navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next_image)
        self.image_slider = QSlider(Qt.Orientation.Horizontal)
        self.image_slider.setMinimum(0)
        self.image_slider.valueChanged.connect(self.slider_value_changed)

        self.image_navigation_layout.addWidget(self.prev_button)
        self.image_navigation_layout.addWidget(self.image_slider)
        self.image_navigation_layout.addWidget(self.next_button)

        self.image_viewer_layout.addLayout(self.image_navigation_layout)

        layout.addLayout(self.image_viewer_layout)
        self.image_viewer_tab.setLayout(layout)

    def init_dataset_stats_tab(self):
        layout = QVBoxLayout()

        self.stats_layout = QVBoxLayout()
        layout.addLayout(self.stats_layout)

        self.dataset_stats_tab.setLayout(layout)


    def generate_class_colors(self):
        def random_color():
            return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        for class_id in self.class_colors:
            self.class_colors[class_id] = random_color()
        self.update_class_colors_table()

    def on_color_cell_clicked(self, item):
        try:
            if item.column() == 1:  # Check if the clicked cell is in the color column
                row = item.row()
                class_label = self.class_colors_table.item(row, 0).text()
                
                # Find the class_id that corresponds to this label
                class_id = None
                for cid, label in self.id_to_label.items():
                    if str(label) == class_label:
                        class_id = cid
                        break
                
                # If we couldn't find the ID through the mapping, 
                # the label might be the raw class ID itself
                if class_id is None:
                    class_id = class_label
                    
                # Make sure the class_id exists in class_colors
                if class_id in self.class_colors:
                    color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
                    if color.isValid():
                        self.class_colors[class_id] = color
                        self.update_class_colors_table()
                        self.show_image()
        except Exception as e:
            print(f"Error in on_color_cell_clicked: {str(e)}")
            pass

    def update_class_colors_table(self):
        self.class_colors_table.setRowCount(len(self.class_colors))
        row = 0
        for class_id, color in self.class_colors.items():
            # Get label for class ID, defaulting to class_id itself if not found
            class_label = self.id_to_label.get(class_id, class_id)
            
            class_item = QTableWidgetItem(str(class_label))
            color_item = QTableWidgetItem()
            color_item.setBackground(color)
            self.class_colors_table.setItem(row, 0, class_item)
            self.class_colors_table.setItem(row, 1, color_item)
            row += 1

    def change_class_color(self):
        selected_items = self.class_colors_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a class to change its color.")
            return

        class_item = selected_items[0]
        class_id = class_item.text()

        color = QColorDialog.getColor(self.class_colors[class_id], self, "Choose Class Color")
        if color.isValid():
            self.class_colors[class_id] = color
            self.update_class_colors_table()
            self.show_image()

    def get_dataset_stats(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select the dataset root.")
            return

        self.clear_layout(self.stats_layout)

        # Reset class mappings
        self.id_to_label = {}
        self.label_to_id = {}
        
        class_counter = Counter()
        image_counter = 0
        instance_counter = 0

        for label_path in self.label_paths.values():
            if os.path.exists(label_path):
                image_counter += 1
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        class_id = line.strip().split()[0]
                        class_counter[class_id] += 1
                        instance_counter += 1

        total_classes = len(class_counter)
        total_instances = instance_counter
        avg_instances_per_image = total_instances / image_counter if image_counter else 0

        stats_label = QLabel(f"Total Classes: {total_classes}")
        self.stats_layout.addWidget(stats_label)

        images_label = QLabel(f"Total Images: {image_counter}")
        self.stats_layout.addWidget(images_label)

        instances_label = QLabel(f"Total Instances: {total_instances}")
        self.stats_layout.addWidget(instances_label)

        avg_instances_label = QLabel(f"Average Instances per Image: {avg_instances_per_image:.2f}")
        self.stats_layout.addWidget(avg_instances_label)

        # Create a list of class names, using YAML labels if available
        class_names = []
        for class_id in class_counter.keys():
            if hasattr(self, 'yaml_labels') and self.yaml_labels:
                try:
                    # Try to convert numeric class_id to YAML label
                    label = self.yaml_labels[int(class_id)] if int(class_id) < len(self.yaml_labels) else class_id
                    class_names.append(label)
                    self.id_to_label[class_id] = label
                    self.label_to_id[label] = class_id
                    
                except (ValueError, IndexError):
                    class_names.append(class_id)
                    self.id_to_label[class_id] = class_id
                    self.label_to_id[class_id] = class_id
            else:
                class_names.append(class_id)
                self.id_to_label[class_id] = class_id
                self.label_to_id[class_id] = class_id

        class_table = QTableWidget()
        class_table.setColumnCount(3)
        class_table.setHorizontalHeaderLabels(["Class", "Instances", "Percentage"])
        class_table.setRowCount(len(class_counter))
        class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, (class_id, class_name) in enumerate(zip(class_counter.keys(), class_names)):
            count = class_counter[class_id]
            class_item = QTableWidgetItem(str(class_name))
            count_item = QTableWidgetItem(str(count))
            percentage_item = QTableWidgetItem(f"{(count / total_instances) * 100:.2f}%")
            class_table.setItem(row, 0, class_item)
            class_table.setItem(row, 1, count_item)
            class_table.setItem(row, 2, percentage_item)

        self.stats_layout.addWidget(class_table)

        # Dark mode
        plt.style.use('dark_background')

        # Plotting a bar graph for class distribution
        fig, ax = plt.subplots(constrained_layout=True)
        
        # Use YAML labels or fall back to numeric class IDs
        classes = class_names
        counts = list(class_counter.values())

        # Assign colors to classes - ensure all class IDs from counter have colors
        self.class_colors = {class_id: QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                            for class_id in class_counter.keys()}
        
        # Adjust the colors list to match class_counter.keys() order
        colors = [self.class_colors[class_id] for class_id in class_counter.keys()]

        bars = ax.bar(classes, counts, color=[self.rgb_to_hex(c) for c in colors])
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Instances')
        ax.set_title('Class Distribution')
        plt.xticks(rotation=45, size=8, ha='right')
        plt.tight_layout()

        
        # give 10% headroom so counts don’t get clipped
        ymax = max(counts) * 1.1
        ax.set_ylim(0, ymax)

        # Add text labels above bars
        for bar, count in zip(bars, counts):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                yval + (ymax * 0.01),        # little offset above bar
                int(count),
                ha='center',
                va='bottom',
                color='white'
            )

        canvas = FigureCanvas(fig)
        self.stats_layout.addWidget(canvas)

        self.update_class_colors_table()  # Populate the colors table after generating dataset stats

    def rgb_to_hex(self, qcolor):
        return '#{:02x}{:02x}{:02x}'.format(qcolor.red(), qcolor.green(), qcolor.blue())

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def create_slider(self):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setEnabled(False)
        value_label = CustomLineEdit("50")
        value_label.setFixedWidth(40)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setEnabled(False)
        slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
        value_label.textChanged.connect(lambda text, sld=slider: sld.setValue(int(text)) if text.isdigit() else None)
        return slider, value_label

    def add_slider_to_layout(self, layout, label_text, slider, value_label, row):
        label = QLabel(label_text)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

    def select_dataset_root(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if dir_name:
            self.dataset_root = dir_name
            self.dataset_label.setText(dir_name)
            if not self.output_dir_set:
                self.prompt_for_output_dir()
            self.update_sliders_state()
            self.scan_folders()
            self.get_dataset_stats()  # Generate dataset stats when dataset is loaded

    def select_overlay_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Overlay Image Directory")
        if dir_name:
            self.overlay_image_dir = dir_name
            self.overlay_label.setText(dir_name)
            self.update_sliders_state()

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_dir = dir_name
            self.output_dir_label.setText(dir_name)
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
                dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
                if dir_name:
                    self.output_dir = dir_name
                    self.output_dir_label.setText(dir_name)
            else:
                self.output_dir = self.dataset_root + "_Augmented"
                self.output_dir_label.setText(self.output_dir)
            self.output_dir_set = True

    def scan_folders(self):
        # Clear previous data structures
        for key in self.skip_augmentations.keys():
            self.skip_augmentations[key] = []
        
        # Clear class mappings
        self.id_to_label = {}
        self.label_to_id = {}
        self.class_colors = {}

        # Scan dataset for folders and images
        folders = set()
        self.image_paths = []  # Reset image paths
        self.label_paths = {}  # Reset label paths

        for root, dirs, files in os.walk(self.dataset_root):
            if os.path.basename(root).lower() not in ['images', 'labels']:
                for name in dirs:
                    if name.lower() not in ['train', 'val', 'labels', 'images']:
                        folders.add(name)
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        self.image_paths.append(image_path)
                        label_path = os.path.join(self.dataset_root, 'labels', os.path.relpath(image_path, os.path.join(self.dataset_root, 'images')).replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                        self.label_paths[image_path] = label_path

        folders = list(folders)
        folders.sort()

        self.skip_table.setRowCount(len(folders))

        self.folder_list.clear()
        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            folder_item.setFlags(folder_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make folder names read-only
            self.skip_table.setItem(row, 0, folder_item)
            list_item = QListWidgetItem(folder)
            self.folder_list.addItem(list_item)
            for col in range(1, 7):  # Update the range to include the new column
                checkbox = QCheckBox()
                checkbox.setStyleSheet("margin-left: 0px; margin-right: auto;")  # Align checkbox to the left 
                if col == 5:
                    checkbox.setEnabled(False)
                if col == 6:  # Connect the new checkbox to the slot
                    checkbox.stateChanged.connect(lambda state, r=row: self.toggle_skip_all(state, r))
                self.skip_table.setCellWidget(row, col, checkbox)

        # Sort images numerically
        self.image_paths.sort(key=self.natural_keys)

        # Parse YAML labels if available
        yaml_labels = parse_dataset_yaml(self.dataset_root)
        self.yaml_labels = yaml_labels

        # Generate dataset stats which will also initialize class_colors
        self.get_dataset_stats()


    def toggle_skip_all(self, state, row):
        skip_all_checked = state == Qt.CheckState.Checked
        for col in range(1, 6):  # Update to check relevant columns
            checkbox = self.skip_table.cellWidget(row, col)
            checkbox.setEnabled(not skip_all_checked)
        if not self.overlay_image_dir:
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            overlay_checkbox.setEnabled(False)

    def display_images(self, item):
        current_show_original = self.show_original  # Store the current state
        
        self.folder_name = item.text()
        self.folder_images = [
            path for path in self.image_paths 
            if os.path.basename(os.path.dirname(path)) == self.folder_name
        ]
        self.image_slider.setMaximum(len(self.folder_images) - 1)
        self.current_image_index = 0
        self.augmented_image = None
        
        # Restore the state before showing the image
        self.show_original = current_show_original
        self.show_image()

    
    def show_image(self):
        try:
            if self.augmented_image is not None:
                if not self.show_original:
                    self.image_name_label.setText(f"(Preview) {os.path.basename(self.current_image_path)}")
                    self.display_image_and_polygons(self.augmented_image, self.augmented_polygons)
                    self.show_original_btn.setText("Show Original Image")
                else:
                    self.image_name_label.setText(f"(Original) {os.path.basename(self.current_image_path)}")
                    original_image = cv2.imread(self.current_image_path)
                    polygons, _ = self.load_polygons_and_labels(self.label_paths.get(self.current_image_path), original_image.shape)
                    self.display_image_and_polygons(original_image, polygons)
                    self.show_original_btn.setText("Show Augmented Image")
            elif self.folder_images:
                self.current_image_path = self.folder_images[self.current_image_index]
                
                # Use image cache
                image = self.image_cache.get_image(self.current_image_path)
                if image is None:
                    return
                    
                # Check for augmented version
                relative_image_path = os.path.relpath(self.current_image_path, self.dataset_root)
                augmented_image_path = os.path.join(self.output_dir, relative_image_path)

                if os.path.exists(augmented_image_path):
                    if not self.show_original:
                        image = self.image_cache.get_image(augmented_image_path)
                        self.image_name_label.setText(f"(Augmented) {os.path.basename(self.current_image_path)}")
                        self.show_original_btn.setText("Show Original Image")
                    else:
                        self.image_name_label.setText(f"(Original) {os.path.basename(self.current_image_path)}")
                        self.show_original_btn.setText("Show Augmented Image")
                    self.show_original_btn.setEnabled(True)
                else:
                    self.image_name_label.setText(os.path.basename(self.current_image_path))
                    self.show_original_btn.setText("Show Original Image")
                    self.show_original_btn.setEnabled(False)
                    self.show_original = False  # Reset state if no augmented version exists

                # Process labels
                label_path = self.label_paths.get(self.current_image_path)
                if self.show_original:
                    polygons, labels = self.load_polygons_and_labels(label_path, image.shape)
                else:
                    relative_label_path = os.path.relpath(label_path, self.dataset_root)
                    augmented_label_path = os.path.join(self.output_dir, relative_label_path)
                    if os.path.exists(augmented_label_path):
                        polygons, labels = self.load_polygons_and_labels(augmented_label_path, image.shape)
                    else:
                        polygons, labels = self.load_polygons_and_labels(label_path, image.shape)

                self.display_image_and_polygons(image, polygons)
                self.update_navigation_buttons()
                
        except Exception as e:
            print(f"Error showing image: {str(e)}")

    def show_original_image(self):
        if self.current_image_path:
            self.show_original = not self.show_original  # Toggle the state
            self.show_image()  # Refresh the display with the new state


    def update_sliders_state(self):
        enable_normal_sliders = bool(self.dataset_root)
        enable_overlay_sliders = bool(self.overlay_image_dir)
        
        # Enable/disable normal sliders
        self.mirror_slider.setEnabled(enable_normal_sliders)
        self.mirror_value.setEnabled(enable_normal_sliders)
        self.crop_slider.setEnabled(enable_normal_sliders)
        self.crop_value.setEnabled(enable_normal_sliders)
        self.zoom_slider.setEnabled(enable_normal_sliders)
        self.zoom_value.setEnabled(enable_normal_sliders)
        self.rotate_slider.setEnabled(enable_normal_sliders)
        self.rotate_value.setEnabled(enable_normal_sliders)
        self.rotation_random_vs_90_slider.setEnabled(enable_normal_sliders)
        self.rotation_random_vs_90_value.setEnabled(enable_normal_sliders)
        self.zoom_in_vs_out_slider.setEnabled(enable_normal_sliders)
        self.zoom_in_vs_out_value.setEnabled(enable_normal_sliders)
        self.maintain_aspect_ratio_slider.setEnabled(enable_normal_sliders)
        self.maintain_aspect_ratio_value.setEnabled(enable_normal_sliders)
        
        # Enable/disable overlay sliders
        self.overlay_slider.setEnabled(enable_overlay_sliders)
        self.overlay_value.setEnabled(enable_overlay_sliders)

        for row in range(self.skip_table.rowCount()):
            overlay_checkbox = self.skip_table.cellWidget(row, 5)
            overlay_checkbox.setEnabled(enable_overlay_sliders)

    def get_skip_augmentations(self):
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

    def toggle_bounding_boxes(self):
        self.show_bounding_boxes = self.bbox_checkbox.isChecked()
        self.show_image()

    def toggle_polygons(self):
        self.show_polygons = self.polygons_checkbox.isChecked()
        self.show_image()

    def toggle_labels(self):
        self.show_labels = self.labels_checkbox.isChecked()
        self.show_image()

    def toggle_points(self):
        self.show_points = self.points_checkbox.isChecked()
        self.show_image()

    def get_augmentation_order(self):
        """Get the current order of augmentations from the sliders list"""
        augmentation_order = []
        valid_augmentation_types = ["mirror", "crop", "zoom", "rotate", "overlay"]
        
        for i in range(self.sliders_list.count()):
            item_widget = self.sliders_list.itemWidget(self.sliders_list.item(i))
            for j in range(item_widget.layout().count()):
                widget = item_widget.layout().itemAt(j).widget()
                if isinstance(widget, QSlider):
                    for attr_name, attr_value in vars(self).items():
                        if attr_value is widget and attr_name in self.slider_to_augmentation_type:
                            aug_type = self.slider_to_augmentation_type[attr_name]
                            if aug_type in valid_augmentation_types:
                                augmentation_order.append(aug_type)
                            break
                    break
        
        return augmentation_order

    def display_image_and_polygons(self, image, polygons):
        for polygon in polygons:
            class_id = polygon[0]
            if class_id not in self.class_colors:
                self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        height, width, _ = image.shape
        image_bytes = image.tobytes()
        qimage = QImage(image_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        qimage = qimage.rgbSwapped()

        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(scaled_pixmap)

        # Enable anti-aliasing for sharper lines and text
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)


        labels = []  # To store labels and positions for later drawing

        if self.augmented_image is not None and self.augmented_image_original_dims is not None:
            orig_h, orig_w = self.augmented_image_original_dims
        else:
            orig_h, orig_w = height, width

        for polygon in polygons:
            class_id = polygon[0]
            if class_id not in self.class_colors:
                self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Set pen for polygon lines and bounding boxes to full opacity
            pen = QPen(self.class_colors[class_id], 2)
            pen_color = QColor(self.class_colors[class_id].red(), self.class_colors[class_id].green(), self.class_colors[class_id].blue(), 255) # Full opacity
            pen.setColor(pen_color)

            # Set brush color with desired opacity
            brush_color = QColor(self.class_colors[class_id].red(), self.class_colors[class_id].green(), self.class_colors[class_id].blue(), 100)  # Set fill opacity here (0-255)
            brush = QBrush(brush_color)
            brush.setStyle(Qt.BrushStyle.SolidPattern)

            points = [QPointF(pt[0] * scaled_pixmap.width() / orig_w, pt[1] * scaled_pixmap.height() / orig_h) for pt in polygon[1:]]

            if self.show_polygons:
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPolygon(*points)

            if self.show_points:
                for point in points:
                    painter.setPen(QPen(Qt.GlobalColor.black, 1))
                    painter.drawEllipse(point, 2.5, 2.5)
                    painter.setPen(QPen(pen_color, 1))
                    painter.drawEllipse(point, 1.5, 1.5)

            # Calculate bounding box
            min_x = min(point.x() for point in points)
            max_x = max(point.x() for point in points)
            min_y = min(point.y() for point in points)
            max_y = max(point.y() for point in points)

            if self.show_bounding_boxes:
                # Draw bounding box with full opacity
                bounding_box_pen = QPen(pen_color, 1)
                painter.setPen(bounding_box_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))

            # Store label information for later drawing
            labels.append((class_id, points[0]))

        # Draw all labels
        if self.show_labels:
            for class_id, position in labels:
                # Check if we have a YAML label for this class
                display_label = class_id
                if hasattr(self, 'yaml_labels') and self.yaml_labels:
                    try:
                        # Try to convert numeric class_id to YAML label
                        display_label = self.yaml_labels[int(class_id)] if int(class_id) < len(self.yaml_labels) else class_id
                    except (ValueError, IndexError):
                        # If conversion fails, keep the original class_id
                        pass

                # Draw text with black outline
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                font_metrics = painter.fontMetrics()
                text_height = font_metrics.height()
                text_width = font_metrics.horizontalAdvance(display_label)
                
                # Adjust label position to prevent going off the canvas
                label_x = position.x()
                label_y = position.y() + text_height

                # Ensure X-coordinate is within canvas
                label_x = max(0, min(label_x, scaled_pixmap.width() - text_width))
                
                # Ensure Y-coordinate is within canvas
                label_y = max(text_height, min(label_y, scaled_pixmap.height() - 5))

                label_position = QPointF(label_x, label_y)
                
                # Draw black outline
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    painter.drawText(label_position + QPointF(dx, dy), display_label)

                # Draw text in white on top
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(label_position, display_label)

        painter.end()

        self.image_label.setPixmap(scaled_pixmap)

    def convert_bbox_to_polygon(self, bbox):
        class_id = bbox[0]
        x_center, y_center, width, height = map(float, bbox[1:])
        half_w = width / 2
        half_h = height / 2
        points = [
            x_center - half_w, y_center - half_h,
            x_center + half_w, y_center - half_h,
            x_center + half_w, y_center + half_h,
            x_center - half_w, y_center + half_h
        ]
        return [class_id] + points

    def load_polygons_and_labels(self, label_path, target_size):
        polygons = []

        if (label_path and os.path.exists(label_path)):
            with open(label_path, 'r') as f:
                label_data = f.readlines()

            for line in label_data:
                line_data = line.strip().split()
                annotation_type = self.identify_annotation_type(line_data)
                if (annotation_type == 'bbox'):
                    polygon_data = self.convert_bbox_to_polygon(line_data)
                else:
                    polygon_data = line_data

                if len(polygon_data) < 5:
                    continue  # Ensure there are enough coordinates for a polygon

                class_id = str(polygon_data[0])  # Convert to string here
                if class_id not in self.class_colors:
                    self.class_colors[class_id] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                pen = QPen(self.class_colors[class_id], 2)
                brush_color = QColor(self.class_colors[class_id].red(), self.class_colors[class_id].green(), self.class_colors[class_id].blue(), 100)  # Set opacity here (0-255)
                brush = QBrush(brush_color)
                brush.setStyle(Qt.BrushStyle.SolidPattern)

                # Extract normalized coordinates
                coords = list(map(float, polygon_data[1:]))
                points = [(coords[i] * target_size[1], coords[i+1] * target_size[0]) for i in range(0, len(coords), 2)]

                polygons.append([class_id] + points)

        return polygons, []

    def identify_annotation_type(self, parts):
        
        if len(parts) < 5:
            return "unknown"
        if len(parts) % 2 == 1 and len(parts) > 5:
            return "polygon"
        elif len(parts) == 5:
            return "bbox"
        else:
            return "unknown"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()

    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.folder_images) - 1)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_slider.setValue(self.current_image_index)
            self.augmented_image = None
            self.show_image()

    def show_next_image(self):
        if self.current_image_index < len(self.folder_images) - 1:
            self.current_image_index += 1
            self.image_slider.setValue(self.current_image_index)
            self.augmented_image = None
            self.show_image()

    def slider_value_changed(self, value):
        if value != self.current_image_index:
            self.current_image_index = value
            self.augmented_image = None
            self.show_image()

    def run_augmentation(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select the dataset root.")
            return

        if not self.output_dir_set:
            self.prompt_for_output_dir()

        # Create progress dialog without cancel button
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setCancelButton(None)  # Remove the default cancel button
        self.progress_dialog.setWindowTitle("Processing Images")
        self.progress_dialog.setMinimumWidth(600)
        self.progress_dialog.setMinimumHeight(300)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        
        # Create layout for the progress dialog
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

        # Remove the default label from QProgressDialog
        self.progress_dialog.findChild(QLabel).hide()
        
        # Make the dialog modal
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        
        # Get the current augmentation order
        augmentation_order = self.get_augmentation_order()

        # Prepare parameters
        params = {
            'image_dir': os.path.join(self.dataset_root, 'images'),
            'label_dir': os.path.join(self.dataset_root, 'labels'),
            'augmented_image_dir': os.path.join(self.output_dir, 'images'),
            'augmented_label_dir': os.path.join(self.output_dir, 'labels'),
            'skip_existing': self.skip_existing_checkbox.isChecked(),
            'skip_augmentations': self.get_skip_augmentations(),
            'mirror_weights': [self.mirror_slider.value(), 100 - self.mirror_slider.value()],
            'crop_weights': [self.crop_slider.value(), 100 - self.crop_slider.value()],
            'zoom_weights': [self.zoom_slider.value(), 100 - self.zoom_slider.value()],
            'rotate_weights': [self.rotate_slider.value(), 100 - self.rotate_slider.value()],
            'overlay_weights': ([self.overlay_slider.value(), 100 - self.overlay_slider.value()] 
                            if self.overlay_image_dir else [0, 100]),
            'rotation_random_vs_90_weights': [self.rotation_random_vs_90_slider.value(), 
                                            100 - self.rotation_random_vs_90_slider.value()],
            'overlay_min_max_scale': self.overlay_min_max_scale,
            'maintain_aspect_ratio_weights': [self.maintain_aspect_ratio_slider.value(),
                                            100 - self.maintain_aspect_ratio_slider.value()],
            'zoom_in_vs_out_weights': [self.zoom_in_vs_out_slider.value(),
                                    100 - self.zoom_in_vs_out_slider.value()],
            'zoom_padding': self.zoom_padding,
            'coco_image_folder': self.overlay_image_dir if self.overlay_image_dir else "",
            'augmentation_order': augmentation_order
        }

        # Create and configure worker
        self.worker = AugmentationWorker(params)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.progress.connect(self.update_progress)  
        self.worker.progress_log.connect(lambda msg: self.log_text.append(msg))
        self.worker.finished.connect(self.handle_completion)
        self.worker.error.connect(self.handle_augmentation_error)
        
        # Initial time for ETA calculation
        self.start_time = time.time()
        self.last_time_update = self.start_time
        self.is_cancelled = False
        
        # Start processing
        self.worker.start()
        self.progress_dialog.exec()  # Show the dialog and wait for completion

    def update_progress(self, value):
        """Update progress and time estimates"""
        if not self.is_cancelled and value > 0:
            try:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # Always update the elapsed time counter
                elapsed_str = self.format_elapsed_time(elapsed_time)
                self.elapsed_time_label.setText(elapsed_str)
                
                # Always update the progress percentage
                self.progress_label.setText(f"Progress: {value}%")
                
                # Only update the estimated remaining time every second
                if current_time - self.last_time_update >= 1.0:
                    estimated_total_time = elapsed_time * 100 / value
                    remaining_time = estimated_total_time - elapsed_time
                    
                    # Format time remaining
                    if remaining_time < 60:
                        time_str = f"{int(remaining_time)} seconds"
                    elif remaining_time < 3600:
                        time_str = f"{int(remaining_time / 60)} minutes"
                    else:
                        time_str = f"{remaining_time / 3600:.1f} hours"
                    
                    # Update remaining time estimate
                    self.time_label.setText(f"Estimated remaining: {time_str}")
                    
                    # Update the last update time
                    self.last_time_update = current_time
                    
            except RuntimeError:
                # Widget has been deleted, ignore the update
                pass

    def handle_completion(self):
        """Handle successful completion of the augmentation process"""
        if not self.is_cancelled:
            # Clear the image cache so that re-displayed images are freshly loaded
            self.image_cache.clear()

            # Close the progress dialog
            self.progress_dialog.close()
            
            # Calculate total elapsed time
            elapsed_time = self.worker.end_time - self.worker.start_time
            formatted_time = self.format_elapsed_time(elapsed_time)
            
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


    def handle_cancellation(self):
        """Handle user cancellation of the augmentation process"""
        self.is_cancelled = True
        self.worker.cancel()
        self.log_text.append("\nCancelling...")
        self.progress_dialog.close()
        
    def handle_augmentation_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during augmentation: {error_msg}")

    def format_elapsed_time(self, seconds):
        """Format elapsed time into HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:d}:{secs:02d}"
        
    def augment_current_image(self):
        if not self.dataset_root:
            QMessageBox.warning(self, "Input Required", "Please select an image to augment.")
            return
        
        mirror_weights = [self.mirror_slider.value(), 100 - self.mirror_slider.value()]
        crop_weights = [self.crop_slider.value(), 100 - self.crop_slider.value()]
        zoom_weights = [self.zoom_slider.value(), 100 - self.zoom_slider.value()]
        rotate_weights = [self.rotate_slider.value(), 100 - self.rotate_slider.value()]
        maintain_aspect_ratio_weights = [self.maintain_aspect_ratio_slider.value(), 100 - self.maintain_aspect_ratio_slider.value()]
        zoom_in_vs_out_weights = [self.zoom_in_vs_out_slider.value(), 100 - self.zoom_in_vs_out_slider.value()]
        rotation_random_vs_90_weights = [self.rotation_random_vs_90_slider.value(), 100 - self.rotation_random_vs_90_slider.value()]
        overlay_weights = [self.overlay_slider.value(), 100 - self.overlay_slider.value()] if self.overlay_image_dir else [0, 100]

        # Get the current augmentation order
        augmentation_order = self.get_augmentation_order()
        
        # Load the image
        image = cv2.imread(self.current_image_path)
        (h, w) = image.shape[:2]

        # Get a random overlay image if available
        overlay_image = None
        if self.overlay_image_dir and os.path.exists(self.overlay_image_dir):
            overlay_files = [f for f in os.listdir(self.overlay_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if overlay_files:
                random_file = random.choice(overlay_files)
                overlay_path = os.path.join(self.overlay_image_dir, random_file)
                overlay_image = cv2.imread(overlay_path)

        # Load the label file if it exists
        label_path = self.label_paths.get(self.current_image_path)
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                polygons = []
                class_ids = []
                for line in lines:
                    parts = line.strip().split()
                    if self.identify_annotation_type(parts) == 'bbox':
                        parts = self.convert_bbox_to_polygon(parts)
                    
                    class_id = parts[0]
                    class_ids.append(class_id)
                    polygon = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
                    polygons.append(polygon)
        else:
            polygons = []
            class_ids = []

        # Run the augment_image function
        augmented_image, augmented_polygons = augment_image(
            image,
            polygons,
            self.folder_name,
            class_ids,
            h,
            w,
            self.skip_augmentations, 
            mirror_weights, 
            crop_weights,
            overlay_weights, 
            rotate_weights,
            rotation_random_vs_90_weights,
            self.overlay_min_max_scale,
            maintain_aspect_ratio_weights, 
            zoom_weights, 
            zoom_in_vs_out_weights,
            self.zoom_padding,
            overlay_image,  # Pass the image directly instead of the folder path
            augmentation_order=augmentation_order
        )

        (new_h, new_w) = augmented_image.shape[:2]

        denormalized_polygons = []
        for polygon in augmented_polygons:
            class_id = polygon[0]
            denormalized_polygon = [class_id] + [(int(x * new_w), int(y * new_h)) for (x, y) in polygon[1:]]
            denormalized_polygons.append(denormalized_polygon)

        self.augmented_image = augmented_image
        self.augmented_polygons = denormalized_polygons
        self.augmented_image_original_dims = (new_h, new_w)
        self.show_image()

    def augment_and_save_current_image(self):
        self.augment_current_image()
        self.save_augmented_image()

    def save_current_preview(self):
        if self.augmented_image is None or not self.augmented_image.any():
            QMessageBox.warning(self, "No Augmented Image", "There is no augmented image to save.")
            return
        
        # Always force overwrite when saving preview
        self.save_augmented_image(force_overwrite=True)

    def save_augmented_image(self, force_overwrite=False):
        if not self.output_dir_set:
            self.prompt_for_output_dir()

        if self.augmented_image is not None:
            relative_image_path = os.path.relpath(self.current_image_path, self.dataset_root)
            augmented_image_path = os.path.join(self.output_dir, relative_image_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)

            # Always overwrite if force_overwrite is True
            if force_overwrite:
                result = cv2.imwrite(augmented_image_path, self.augmented_image)
                if not result:
                    QMessageBox.warning(self, "Save Error", f"Failed to save image at {augmented_image_path}")
            elif not os.path.exists(augmented_image_path):
                result = cv2.imwrite(augmented_image_path, self.augmented_image)
                if not result:
                    QMessageBox.warning(self, "Save Error", f"Failed to save image at {augmented_image_path}")
            else:
                QMessageBox.information(self, "Skip", f"Image already exists and was skipped: {augmented_image_path}")

            # Save augmented label
            relative_label_path = os.path.relpath(self.label_paths[self.current_image_path], self.dataset_root)
            augmented_label_path = os.path.join(self.output_dir, relative_label_path)

            os.makedirs(os.path.dirname(augmented_label_path), exist_ok=True)
            with open(augmented_label_path, 'w') as f:
                for polygon in self.augmented_polygons:
                    class_id = polygon[0]
                    coords = [f"{x / self.augmented_image_original_dims[1]} {y / self.augmented_image_original_dims[0]}" for x, y in polygon[1:]]
                    f.write(f"{class_id} {' '.join(coords)}\n")

            # Invalidate cache to force reload
            if augmented_image_path in self.image_cache.cache:
                del self.image_cache.cache[augmented_image_path]



    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AugmentDatasetGUI()
    ex.show()
    sys.exit(app.exec())