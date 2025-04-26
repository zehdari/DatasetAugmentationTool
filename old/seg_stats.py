import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class SegmentationStatsAnalyzer:
    """
    Utility class for loading segmentation annotations and generating
    an interactive Matplotlib pairplot where clicking the legend
    entries hides/shows the original vs. augmented data.
    """
    @staticmethod
    def load_annotations(dataset_folder: str) -> pd.DataFrame:
        data = {'x_center': [], 'y_center': [], 'width': [], 'height': [], 'area': []}
        for split in ['train', 'val']:
            split_folder = os.path.join(dataset_folder, 'labels', split)
            if not os.path.isdir(split_folder):
                continue
            for folder_name in os.listdir(split_folder):
                folder_path = os.path.join(split_folder, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                for fname in os.listdir(folder_path):
                    if not fname.endswith('.txt'):
                        continue
                    path = os.path.join(folder_path, fname)
                    with open(path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            coords = list(map(float, parts[1:]))
                            xs, ys = coords[0::2], coords[1::2]
                            xmin, xmax = min(xs), max(xs)
                            ymin, ymax = min(ys), max(ys)
                            w, h = xmax - xmin, ymax - ymin
                            xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
                            # Shoelace formula (polygon area)
                            area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
                            data['x_center'].append(xc)
                            data['y_center'].append(yc)
                            data['width'].append(w)
                            data['height'].append(h)
                            data['area'].append(area)
        return pd.DataFrame(data)

    @staticmethod
    def popup_pairplot_compare(
        df_orig: pd.DataFrame,
        df_aug: pd.DataFrame,
        show_orig: bool = True,
        show_aug: bool = True
    ) -> plt.Figure:
        """
        Build a Seaborn pairplot comparing original vs. augmented annotations.
        Clicking the legend entries toggles visibility of each dataset.

        Parameters:
        - df_orig: DataFrame of original annotations
        - df_aug:  DataFrame of augmented annotations
        - show_orig: include original data if True
        - show_aug:  include augmented data if True

        Returns:
        - Matplotlib Figure containing the pairplot. No plt.show() is called,
          so you can embed this in your application's FigureCanvas.
        """
        parts = []
        if show_orig and df_orig is not None and not df_orig.empty:
            o = df_orig.copy()
            o['source'] = 'original'
            parts.append(o)
        if show_aug and df_aug is not None and not df_aug.empty:
            a = df_aug.copy()
            a['source'] = 'augmented'
            parts.append(a)
        if not parts:
            raise ValueError("No data to plot: ensure at least one of original or augmented is available.")

        combined = pd.concat(parts, ignore_index=True)

        # Create the pairplot
        grid = sns.pairplot(
            combined,
            hue='source',
            kind='scatter',
            diag_kind='hist',
            plot_kws={'alpha': 0.3},
            diag_kws={'alpha': 0.5},
            corner=False
        )
        grid.fig.suptitle('Comparison Pairplot: Original vs Augmented', y=1.02)
        grid.fig.tight_layout()

        # Make legend items pickable
        legend = grid._legend
        for handle in legend.legend_handles:
            handle.set_picker(True)

        def on_pick(event):
            handle = event.artist
            label = handle.get_label()
            # Toggle legend marker alpha
            current_alpha = handle.get_alpha() or 1.0
            handle.set_alpha(1.0 if current_alpha < 1.0 else 0.2)

            # Toggle visibility of matching artists
            for ax in grid.axes.flatten():
                for coll in ax.collections:  # scatter dots
                    if coll.get_label() == label:
                        coll.set_visible(not coll.get_visible())
                for patch in ax.patches:    # histogram bars
                    if patch.get_label() == label:
                        patch.set_visible(not patch.get_visible())

            # redraw in whatever GUI your app is using:
            grid.fig.canvas.draw_idle()

        grid.fig.canvas.mpl_connect('pick_event', on_pick)

        # Return the figure; caller can embed or show it as needed
        return grid.fig
