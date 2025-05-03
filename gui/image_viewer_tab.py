from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QCheckBox, QSlider, QListWidget, QListWidgetItem, QSizePolicy)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QImage
import cv2
import os
import re
import random

class ImageViewerTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
        # Initialize state variables
        self.image_paths = []
        self.folder_images = []
        self.label_paths = {}
        self.current_image_index = 0
        self.current_image_path = ""
        self.folder_name = ""
        self.augmented_image = None
        self.augmented_polygons = None
        self.augmented_image_original_dims = None
        
        # Display options
        self.show_labels = True
        self.show_polygons = True 
        self.show_bounding_boxes = False 
        self.show_points = False
        self.show_original = False
        
        self.initUI()
        
    def initUI(self):
        self.image_viewer_layout = QVBoxLayout()
        folder_list_layout = QHBoxLayout()
        folder_and_button_layout = QVBoxLayout()

        self.folder_list = QListWidget()
        self.folder_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
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

        self.setLayout(self.image_viewer_layout)

    def display_images(self, item):
        current_show_original = self.show_original  # Store the current state
        
        self.folder_name = item.text()
        self.folder_images = [
            path for path in self.parent.image_paths 
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
                    polygons = self.load_polygons(self.parent.label_paths.get(self.current_image_path), original_image.shape)
                    self.display_image_and_polygons(original_image, polygons)
                    self.show_original_btn.setText("Show Augmented Image")
            elif self.folder_images:
                self.current_image_path = self.folder_images[self.current_image_index]
                
                # Use image cache
                image = self.parent.image_cache.get_image(self.current_image_path)
                if image is None:
                    return
                    
                # Check for augmented version
                relative_image_path = os.path.relpath(self.current_image_path, self.parent.dataset_root)
                augmented_image_path = os.path.join(self.parent.output_dir, relative_image_path)

                if os.path.exists(augmented_image_path):
                    if not self.show_original:
                        image = self.parent.image_cache.get_image(augmented_image_path)
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
                label_path = self.parent.label_paths.get(self.current_image_path)
                if self.show_original:
                    polygons = self.load_polygons(label_path, image.shape)
                else:
                    relative_label_path = os.path.relpath(label_path, self.parent.dataset_root)
                    augmented_label_path = os.path.join(self.parent.output_dir, relative_label_path)
                    if os.path.exists(augmented_label_path):
                        polygons = self.load_polygons(augmented_label_path, image.shape)
                    else:
                        polygons = self.load_polygons(label_path, image.shape)

                self.display_image_and_polygons(image, polygons)
                self.update_navigation_buttons()
                
        except Exception as e:
            print(f"Error showing image: {str(e)}")

    # Add remaining methods needed for the image viewer functionality
    def show_original_image(self):
        if self.current_image_path:
            self.show_original = not self.show_original  # Toggle the state
            self.show_image()

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
        
    def augment_current_image(self):
        if not self.parent.dataset_root or not self.current_image_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Input Required", "Please select an image to augment.")
            return
        
        # Get augmentation parameters from the settings tab
        params = self.parent.settings_tab.get_augmentation_params()
        mirror_weights = params['mirror_weights']
        crop_weights = params['crop_weights']
        zoom_weights = params['zoom_weights']
        rotate_weights = params['rotate_weights']
        maintain_aspect_ratio_weights = params['maintain_aspect_ratio_weights']
        zoom_in_vs_out_weights = params['zoom_in_vs_out_weights']
        rotation_random_vs_90_weights = params['rotation_random_vs_90_weights']
        overlay_weights = params['overlay_weights']
        augmentation_order = params['augmentation_order']
        
        # Extract individual padding parameters
        zoom_in_min_padding = params.get('zoom_in_min_padding', 0.1)
        zoom_in_max_padding = params.get('zoom_in_max_padding', 0.3)
        zoom_out_min_padding = params.get('zoom_out_min_padding', 0.1)
        zoom_out_max_padding = params.get('zoom_out_max_padding', 0.5)
        
        # Extract individual overlay scale parameters
        overlay_min_scale = params.get('overlay_min_scale', 0.3)
        overlay_max_scale = params.get('overlay_max_scale', 1.0)
        
        # Import the augmenter from the core package
        from core.augment_data import ImageAugmenter
        
        # Load the image
        image = cv2.imread(self.current_image_path)

        # Get a random overlay image if available
        overlay_image = None
        if self.parent.overlay_image_dir and os.path.exists(self.parent.overlay_image_dir):
            overlay_files = [f for f in os.listdir(self.parent.overlay_image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if overlay_files:
                random_file = random.choice(overlay_files)
                overlay_path = os.path.join(self.parent.overlay_image_dir, random_file)
                overlay_image = cv2.imread(overlay_path)

        # Load the label file if it exists
        label_path = self.parent.label_paths.get(self.current_image_path)
        polygons = []
        class_ids = []
        
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if self.identify_annotation_type(parts) == 'bbox':
                        parts = self.convert_bbox_to_polygon(parts)
                    
                    class_id = parts[0]
                    class_ids.append(class_id)
                    polygon = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
                    polygons.append(polygon)

        # Initialize the augmenter
        augmenter = ImageAugmenter()
        
        # Run the augment_image function with individual parameters
        augmented_image, augmented_polygons = augmenter.augment_image(
            image=image,
            polygons=polygons,
            current_subfolder=self.folder_name,
            class_ids=class_ids,
            skip_augmentations=params['skip_augmentations'], 
            mirror_weights=mirror_weights, 
            crop_weights=crop_weights,
            overlay_weights=overlay_weights, 
            rotate_weights=rotate_weights,
            rotation_random_vs_90_weights=rotation_random_vs_90_weights,
            maintain_aspect_ratio_weights=maintain_aspect_ratio_weights, 
            zoom_weights=zoom_weights, 
            zoom_in_vs_out_weights=zoom_in_vs_out_weights,
            # Pass individual parameters instead of compound ones
            zoom_in_min_padding=zoom_in_min_padding,
            zoom_in_max_padding=zoom_in_max_padding,
            zoom_out_min_padding=zoom_out_min_padding,
            zoom_out_max_padding=zoom_out_max_padding,
            overlay_min_scale=overlay_min_scale,
            overlay_max_scale=overlay_max_scale,
            coco_image=overlay_image,
            augmentation_order=augmentation_order
        )

        (new_h, new_w) = augmented_image.shape[:2]

        # Convert the augmented polygons back to display format
        denormalized_polygons = []
        for polygon in augmented_polygons:
            class_id = polygon[0]
            denormalized_polygon = [class_id] + [(int(x * new_w), int(y * new_h)) for (x, y) in polygon[1:]]
            denormalized_polygons.append(denormalized_polygon)

        # Store the augmentation results
        self.augmented_image = augmented_image
        self.augmented_polygons = denormalized_polygons
        self.augmented_image_original_dims = (new_h, new_w)
        
        # Display the augmented image
        self.show_image()

    def save_current_preview(self):
        if self.augmented_image is None or not self.augmented_image.any():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Augmented Image", "There is no augmented image to save.")
            return
        
        # Check if output directory is set
        if not self.parent.output_dir_set:
            self.parent.prompt_for_output_dir()

        # Save the augmented image with overwrite
        if self.augmented_image is not None:
            relative_image_path = os.path.relpath(self.current_image_path, self.parent.dataset_root)
            augmented_image_path = os.path.join(self.parent.output_dir, relative_image_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)

            # Always overwrite when saving preview
            result = cv2.imwrite(augmented_image_path, self.augmented_image)
            if not result:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Save Error", f"Failed to save image at {augmented_image_path}")
            else:
                # Save augmented label
                relative_label_path = os.path.relpath(self.parent.label_paths[self.current_image_path], self.parent.dataset_root)
                augmented_label_path = os.path.join(self.parent.output_dir, relative_label_path)

                os.makedirs(os.path.dirname(augmented_label_path), exist_ok=True)
                with open(augmented_label_path, 'w') as f:
                    for polygon in self.augmented_polygons:
                        class_id = polygon[0]
                        coords = [f"{x / self.augmented_image_original_dims[1]} {y / self.augmented_image_original_dims[0]}" for x, y in polygon[1:]]
                        f.write(f"{class_id} {' '.join(coords)}\n")

                # Invalidate cache to force reload on next view
                if augmented_image_path in self.parent.image_cache.cache:
                    del self.parent.image_cache.cache[augmented_image_path]
                    
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"Saved augmented image and labels to:\n{augmented_image_path}")

    def display_image_and_polygons(self, image, polygons):
        # Make sure we have class colors for each polygon
        for polygon in polygons:
            class_id = polygon[0]
            if class_id not in self.parent.settings_tab.class_colors:
                color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                self.parent.settings_tab.class_colors[class_id] = color
        
        # Create QImage and QPixmap from CV2 image
        height, width, _ = image.shape
        image_bytes = image.tobytes()
        qimage = QImage(image_bytes, width, height, width * 3, QImage.Format.Format_RGB888)
        qimage = qimage.rgbSwapped()

        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
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
            color = self.parent.settings_tab.class_colors[class_id]

            # Set pen for polygon lines and bounding boxes to full opacity
            pen = QPen(color, 2)
            pen_color = QColor(color.red(), color.green(), color.blue(), 255)  # Full opacity
            pen.setColor(pen_color)

            # Set brush color with desired opacity
            brush_color = QColor(color.red(), color.green(), color.blue(), 100)  # Set fill opacity here (0-255)
            brush = QBrush(brush_color)
            brush.setStyle(Qt.BrushStyle.SolidPattern)

            # Convert normalized coordinates to pixel coordinates in the scaled pixmap
            points = [QPointF(pt[0] * scaled_pixmap.width() / orig_w, 
                            pt[1] * scaled_pixmap.height() / orig_h) for pt in polygon[1:]]

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
            if len(points) > 0:
                labels.append((class_id, points[0]))

        # Draw all labels at the end so they're always on top
        if self.show_labels:
            for class_id, position in labels:
                # Check if we have a YAML label for this class
                display_label = class_id
                if hasattr(self.parent, 'stats_tab') and self.parent.stats_tab.yaml_labels:
                    try:
                        # Try to convert numeric class_id to YAML label
                        yaml_labels = self.parent.stats_tab.yaml_labels
                        display_label = yaml_labels[int(class_id)] if int(class_id) < len(yaml_labels) else class_id
                    except (ValueError, IndexError):
                        # If conversion fails, keep the original class_id
                        pass

                # Draw text with black outline
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                font_metrics = painter.fontMetrics()
                text_height = font_metrics.height()
                text_width = font_metrics.horizontalAdvance(str(display_label))
                
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
                    painter.drawText(label_position + QPointF(dx, dy), str(display_label))

                # Draw text in white on top
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(label_position, str(display_label))

        painter.end()
        self.image_label.setPixmap(scaled_pixmap)

    def load_polygons(self, label_path, target_size):
        polygons = []

        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_data = f.readlines()

            for line in label_data:
                line_data = line.strip().split()
                
                # Skip empty lines
                if not line_data:
                    continue
                    
                annotation_type = self.identify_annotation_type(line_data)
                if annotation_type == 'bbox':
                    polygon_data = self.convert_bbox_to_polygon(line_data)
                else:
                    polygon_data = line_data

                if len(polygon_data) < 5:
                    continue  # Ensure there are enough coordinates for a polygon

                class_id = str(polygon_data[0])  # Convert to string
                
                # Add class id to class colors if not present
                if class_id not in self.parent.settings_tab.class_colors:
                    self.parent.settings_tab.class_colors[class_id] = QColor(
                        random.randint(0, 255), 
                        random.randint(0, 255), 
                        random.randint(0, 255)
                    )

                # Extract normalized coordinates
                coords = list(map(float, polygon_data[1:]))
                points = [(coords[i] * target_size[1], coords[i+1] * target_size[0]) for i in range(0, len(coords), 2)]

                polygons.append([class_id] + points)

        return polygons

    def identify_annotation_type(self, parts):
        if len(parts) < 5:
            return "unknown"
        if len(parts) % 2 == 1 and len(parts) > 5:
            return "polygon"
        elif len(parts) == 5:
            return "bbox"
        else:
            return "unknown"

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
        
    # Image navigation methods
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
            
    def update_navigation_buttons(self):
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.folder_images) - 1)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_image()