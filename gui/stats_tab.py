from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget, 
                            QTableWidgetItem, QHeaderView)
from PyQt6.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from collections import Counter
import random
import os
import yaml

class DatasetStatsTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.yaml_labels = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        self.stats_layout = QVBoxLayout()
        layout.addLayout(self.stats_layout)
        self.setLayout(layout)
        
    def get_dataset_stats(self):
        if not self.parent.dataset_root:
            return

        self.clear_layout(self.stats_layout)

        # Reset class mappings
        self.parent.settings_tab.id_to_label = {}
        
        class_counter = Counter()
        image_counter = 0
        instance_counter = 0

        for label_path in self.parent.label_paths.values():
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
            if self.yaml_labels:
                try:
                    # Try to convert numeric class_id to YAML label
                    label = self.yaml_labels[int(class_id)] if int(class_id) < len(self.yaml_labels) else class_id
                    class_names.append(label)
                    self.parent.settings_tab.id_to_label[class_id] = label
                    
                except (ValueError, IndexError):
                    class_names.append(class_id)
                    self.parent.settings_tab.id_to_label[class_id] = class_id
            else:
                class_names.append(class_id)
                self.parent.settings_tab.id_to_label[class_id] = class_id

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

        # Generate class colors
        for class_id in class_counter.keys():
            if class_id not in self.parent.settings_tab.class_colors:
                self.parent.settings_tab.class_colors[class_id] = QColor(
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255)
                )

        # Create visualization
        self.create_class_distribution_chart(class_names, class_counter)
        
        # Update settings tab class colors table
        self.parent.settings_tab.update_class_colors_table()

    def create_class_distribution_chart(self, classes, class_counter):
        # Dark mode
        plt.style.use('dark_background')

        # Plotting a bar graph for class distribution
        fig, ax = plt.subplots()
        
        counts = list(class_counter.values())

        # Get colors in the right order
        colors = [self.rgb_to_hex(self.parent.settings_tab.class_colors[class_id]) 
                 for class_id in class_counter.keys()]

        bars = ax.bar(classes, counts, color=colors)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Instances')
        ax.set_title('Class Distribution')
        plt.xticks(rotation=45, size=8, ha='right')
        plt.tight_layout()

        # Give 10% headroom so counts don't get clipped
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
    
    def rgb_to_hex(self, qcolor):
        return '#{:02x}{:02x}{:02x}'.format(qcolor.red(), qcolor.green(), qcolor.blue())
        
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()