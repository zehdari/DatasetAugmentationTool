import os
import re
import random
import time
import cv2
import concurrent.futures
import gc
import psutil
import hashlib
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal
from services.overlay_provider import OverlayProvider
from core.augment_data import ImageAugmenter

def identify_annotation_type(parts):
    if len(parts) < 5:
        return "unknown"
    if len(parts) % 2 == 1 and len(parts) > 5:
        return "polygon"
    elif len(parts) == 5:
        return "bbox"
    else:
        return "unknown"

def convert_bbox_to_polygon(bbox):
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
        
        # Read and parse labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            polygons = []
            class_ids = []
            for line in lines:
                parts = line.strip().split()

                if identify_annotation_type(parts) == 'bbox':
                    parts = convert_bbox_to_polygon(parts)

                class_id = parts[0]
                class_ids.append(class_id)
                coords = [(float(parts[i]), float(parts[i + 1])) 
                         for i in range(1, len(parts), 2)]
                polygons.append(coords)
        
        # Extract the parameters needed for augmentation
        augmentation_params = {}
        
        # Add all parameters directly to augmentation_params
        for key, value in params.items():
            # Skip specific keys that are not augmentation parameters
            if key in ['image_dir', 'label_dir', 'augmented_image_dir', 'augmented_label_dir', 'coco_image_folder']:
                continue
            augmentation_params[key] = value

        # Perform augmentation
        augmenter = ImageAugmenter()
        augmented_image, augmented_polygons = augmenter.augment_image(
            image=image,
            polygons=polygons,
            current_subfolder=subfolder,
            class_ids=class_ids,
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