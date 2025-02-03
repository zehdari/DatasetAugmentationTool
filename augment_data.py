import os
import cv2
import random
import numpy as np
import math
import shutil
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.ops import unary_union

def mirror_polygon(polygon):
    return [(1 - x, y) for x, y in polygon]

def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))

def calculate_overall_bounding_box(polygons):
    # Unpack all points from the polygons
    all_points = [point for polygon in polygons for point in polygon]
    if not all_points:
        return None

    # Calculate min/max x and y coordinates
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)

    return min_x, min_y, max_x, max_y

def rotate_polygon(polygon, angle, original_center, new_center, original_dims, new_dims):
    angle_rad = math.radians(-angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    rotated_polygon = []

    for x, y in polygon:
        # Denormalize points using original dimensions
        x_denorm = x * original_dims[0]
        y_denorm = y * original_dims[1]

        # Shift to origin based on the original center
        x_shifted = x_denorm - original_center[0]
        y_shifted = y_denorm - original_center[1]

        # Rotate
        x_rotated = x_shifted * cos_angle - y_shifted * sin_angle
        y_rotated = x_shifted * sin_angle + y_shifted * cos_angle

        # Shift back using the new center
        x_new = x_rotated + new_center[0]
        y_new = y_rotated + new_center[1]

        # Renormalize using new dimensions
        x_norm = x_new / new_dims[0]
        y_norm = y_new / new_dims[1]

        # Clamp values to ensure they are normalized
        x_clamped = clamp(x_norm)
        y_clamped = clamp(y_norm)

        rotated_polygon.append((x_clamped, y_clamped))

    return rotated_polygon

def get_rotation_angle(rotate_weights):
    rotation_choice = random.choices([True, False], weights=rotate_weights, k=1)[0]
    # 25% chance to rotate randomly
    if rotation_choice:
        return random.uniform(0, 360)
    # 75% chance to rotate by 0, 90, 180, or 270 degrees
    else:
        return random.choice([0, 90, 180, 270])

def mirror_image(image):
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Determine the rotation matrix and calculate the new bounding dimensions of the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    
    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def zoom_out_image_and_polygons(image, polygons, min_padding=0.1, max_padding=0.5):
    if not polygons:
        return image, polygons  # No action if there are no polygons

    img_height, img_width = image.shape[:2]

    # Independently determine padding for width and height
    padding_x = random.uniform(min_padding, max_padding) + 1  # Random padding for width
    padding_y = random.uniform(min_padding, max_padding) + 1  # Random padding for height

    # Calculate the size of the new canvas without maintaining aspect ratio
    canvas_width = int(img_width * padding_x)
    canvas_height = int(img_height * padding_y)

    # Create a new canvas and fill it with black color
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Calculate the position where the original image will be placed on the canvas
    x_offset = (canvas_width - img_width) // 2
    y_offset = (canvas_height - img_height) // 2

    # Place the original image in the center of the canvas
    canvas[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = image

    # Adjust polygon coordinates according to the new image placement
    adjusted_polygons = []
    for polygon in polygons:
        adjusted_polygon = []
        for x, y in polygon:
            # Denormalize coordinates to the original image size
            denormalized_x = x * img_width
            denormalized_y = y * img_height

            # Translate coordinates by adding the offsets
            translated_x = denormalized_x + x_offset
            translated_y = denormalized_y + y_offset

            # Renormalize coordinates to the new canvas size
            new_x = translated_x / canvas_width
            new_y = translated_y / canvas_height

            # Clamp the values to ensure they remain normalized
            new_x_clamped = clamp(new_x)
            new_y_clamped = clamp(new_y)

            adjusted_polygon.append((new_x_clamped, new_y_clamped))
        adjusted_polygons.append(adjusted_polygon)

    return canvas, adjusted_polygons


def zoom_in_image_and_polygons(image, polygons, min_padding=0.1, max_padding=0.8):
    if not polygons:
        return image, []  # Return the image as is if there are no polygons

    img_height, img_width = image.shape[:2]

    # Calculate the collective bounding box for all polygons
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = calculate_overall_bounding_box(polygons)

    # Apply uniform random padding within specified bounds, ensuring no distortion
    padding = random.uniform(min_padding, max_padding)
    padding_x = padding * (bbox_x_max - bbox_x_min)
    padding_y = padding * (bbox_y_max - bbox_y_min)

    # Calculate padded bounding box, ensuring it stays within [0, 1]
    padded_bbox_x_min = max(bbox_x_min - padding_x, 0)
    padded_bbox_y_min = max(bbox_y_min - padding_y, 0)
    padded_bbox_x_max = min(bbox_x_max + padding_x, 1)
    padded_bbox_y_max = min(bbox_y_max + padding_y, 1)

    # Convert padded bounding box to pixel coordinates for cropping
    crop_x_min = int(padded_bbox_x_min * img_width)
    crop_y_min = int(padded_bbox_y_min * img_height)
    crop_x_max = int(padded_bbox_x_max * img_width)
    crop_y_max = int(padded_bbox_y_max * img_height)

    # Crop the image according to the padded bounding box
    cropped_image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # Resize the cropped image back to the original dimensions to maintain aspect ratio
    zoomed_image = cv2.resize(cropped_image, (img_width, img_height))

    # Adjust polygon coordinates to match the zoomed image
    adjusted_polygons = []
    for polygon in polygons:
        adjusted_polygon = [((x - padded_bbox_x_min) / (padded_bbox_x_max - padded_bbox_x_min),
                             (y - padded_bbox_y_min) / (padded_bbox_y_max - padded_bbox_y_min))
                            for x, y in polygon]
        adjusted_polygons.append(adjusted_polygon)

    return zoomed_image, adjusted_polygons

def crop_image_and_polygons(image, polygons, class_ids):
    try:
        img_height, img_width = image.shape[:2]

        # Calculate the overall bounding box of all polygons
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = calculate_overall_bounding_box(polygons)
        
        # Randomly choose between vertical and horizontal crop
        crop_orientation = random.choice(['vertical', 'horizontal'])

        # Randomly determine the crop percentage (1% to 50%)
        crop_percentage = random.uniform(0.01, 0.5)

        if crop_orientation == 'vertical':
            # Calculate the width of the bounding box and the vertical crop width
            bbox_width = bbox_x_max - bbox_x_min
            crop_width = bbox_width * crop_percentage

            # Determine the vertical crop line within the bounding box based on the chosen percentage
            crop_line_x = bbox_x_min + crop_width
            crop_line = int(crop_line_x * img_width)

            # Compare areas on either side of the vertical crop line and decide which side to keep
            keep = 'left' if (crop_line_x - bbox_x_min) > (bbox_x_max - crop_line_x) else 'right'

            # Crop the image vertically at the calculated line
            cropped_image = image[:, :crop_line] if keep == 'left' else image[:, crop_line:]

            # New image bounds in normalized coordinates, considering the vertical crop
            new_image_box = box(bbox_x_min, 0, crop_line_x, 1) if keep == 'left' else box(crop_line_x, 0, bbox_x_max, 1)

        else:  # Horizontal crop
            # Calculate the height of the bounding box and the horizontal crop height
            bbox_height = bbox_y_max - bbox_y_min
            crop_height = bbox_height * crop_percentage

            # Determine the horizontal crop line within the bounding box based on the chosen percentage
            crop_line_y = bbox_y_min + crop_height
            crop_line = int(crop_line_y * img_height)

            # Compare areas above and below the horizontal crop line and decide which side to keep
            keep = 'top' if (crop_line_y - bbox_y_min) > (bbox_y_max - crop_line_y) else 'bottom'

            # Crop the image horizontally at the calculated line
            cropped_image = image[:crop_line, :] if keep == 'top' else image[crop_line:, :]

            # New image bounds in normalized coordinates, considering the horizontal crop
            new_image_box = box(0, bbox_y_min, 1, crop_line_y) if keep == 'top' else box(0, crop_line_y, 1, bbox_y_max)

        # Adjust polygons and retain class IDs based on the new image bounds after cropping
        adjusted_polygons, retained_class_ids = adjust_polygons_and_class_ids(polygons, class_ids, new_image_box, crop_line_x if crop_orientation == 'vertical' else crop_line_y, keep, crop_orientation)

        return cropped_image, adjusted_polygons, retained_class_ids
    except Exception as e:
        # In case of an error during cropping, return the input image and polygons as they were
        print(f"Error during cropping: {e}. Returning original image and polygons.")
        return image, polygons, class_ids


def adjust_polygons_and_class_ids(polygons, class_ids, new_image_box, crop_line, keep, orientation):
    adjusted_polygons = []
    retained_class_ids = []  # Track class IDs for polygons retained after cropping
    
    for i, polygon in enumerate(polygons):
        # Create a Shapely Polygon and attempt to clean it
        shapely_polygon = Polygon(polygon).buffer(0)

        # If the cleaning resulted in a MultiPolygon, take the union of all parts to get a single Polygon
        if shapely_polygon.is_empty:
            continue
        if isinstance(shapely_polygon, MultiPolygon):
            shapely_polygon = unary_union(shapely_polygon)

        # Perform the intersection with the new image box
        intersected_polygon = shapely_polygon.intersection(new_image_box)

        if not intersected_polygon.is_empty:
            if isinstance(intersected_polygon, MultiPolygon):
                for poly in intersected_polygon.geoms:
                    adjusted_polygons.append([(pt[0], pt[1]) for pt in poly.exterior.coords[:-1]])
                    retained_class_ids.append(class_ids[i])
            else:
                adjusted_polygons.append([(pt[0], pt[1]) for pt in intersected_polygon.exterior.coords[:-1]])
                retained_class_ids.append(class_ids[i])

    # Adjust polygon coordinates based on crop orientation and side to keep
    final_polygons = adjust_polygons_for_crop(adjusted_polygons, crop_line, keep, orientation)

    return final_polygons, retained_class_ids

def adjust_polygons_for_crop(polygons, crop_line, keep, orientation):
    final_polygons = []
    for polygon in polygons:
        if (keep == 'right' and orientation == 'vertical') or (keep == 'bottom' and orientation == 'horizontal'):
            adjusted_polygon = [((x - crop_line) / (1 - crop_line), y) if orientation == 'vertical' else (x, (y - crop_line) / (1 - crop_line)) for x, y in polygon]
        else:
            adjusted_polygon = polygon  # No adjustment needed if keeping left/top
        final_polygons.append(adjusted_polygon)
    return final_polygons
    
def pad_image_and_adjust_polygons(cropped_image, adjusted_polygons, original_dimensions):
    cropped_height, cropped_width = cropped_image.shape[:2]
    original_height, original_width = original_dimensions

    # Calculate padding needed to restore original dimensions
    pad_vertical = (original_height - cropped_height) // 2
    pad_horizontal = (original_width - cropped_width) // 2

    # Pad the cropped image
    padded_image = cv2.copyMakeBorder(cropped_image, pad_vertical, pad_vertical, pad_horizontal, pad_horizontal, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Adjust polygon coordinates
    shifted_polygons = []
    for polygon in adjusted_polygons:
        new_polygon = []
        for x, y in polygon:
            # Clamp, denormalize, translate, and renormalize
            x_clamped, y_clamped = clamp(x), clamp(y)
            abs_x, abs_y = x_clamped * cropped_width, y_clamped * cropped_height
        
            # Careful translation considering padding
            translated_x = abs_x + pad_horizontal if pad_horizontal > 0 else abs_x
            translated_y = abs_y + pad_vertical if pad_vertical > 0 else abs_y

            # Renormalize to the original dimensions, considering the padding might have changed the effective area
            new_x = translated_x / original_width
            new_y = translated_y / original_height

            new_polygon.append((clamp(new_x), clamp(new_y)))
        shifted_polygons.append(new_polygon)

    return padded_image, shifted_polygons

def overlay_detections_on_coco(coco_image, image, detection_polygons, min_scale=0.1, max_scale=1.0):
    adjusted_polygons = []
    
    # Initialize an empty mask for all detections
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for idx, polygon in enumerate(detection_polygons):
        # Convert polygon points to integer coordinates
        polygon_points = np.array([(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(combined_mask, [polygon_points], 255)

    # Create an RGBA image with transparency for all detections
    b, g, r = cv2.split(image)
    alpha_channel = np.zeros_like(b)
    alpha_channel[combined_mask > 0] = 255
    rgba_detection = cv2.merge((b, g, r, alpha_channel))

    # Find the bounding box of the combined mask
    x, y, w, h = cv2.boundingRect(combined_mask)
    cropped_detection = rgba_detection[y:y+h, x:x+w]

    # Scale down if the cropped detection is larger than the COCO image
    scale_factor = min(coco_image.shape[0] / cropped_detection.shape[0], 
                      coco_image.shape[1] / cropped_detection.shape[1])

    if scale_factor < 1.0:
        cropped_detection = cv2.resize(cropped_detection, 
                                     (int(cropped_detection.shape[1] * scale_factor), 
                                      int(cropped_detection.shape[0] * scale_factor)))

    random_scale_factor = random.uniform(min_scale, max_scale)
    new_width = int(cropped_detection.shape[1] * random_scale_factor)
    new_height = int(cropped_detection.shape[0] * random_scale_factor)
    cropped_detection = cv2.resize(cropped_detection, (new_width, new_height))

    # Create an overlay with transparency
    overlay = np.zeros((coco_image.shape[0], coco_image.shape[1], 4), dtype=np.uint8)
    x_offset = random.randint(0, max(coco_image.shape[1] - cropped_detection.shape[1], 1))
    y_offset = random.randint(0, max(coco_image.shape[0] - cropped_detection.shape[0], 1))

    # TEST
    overlay[y_offset:y_offset + cropped_detection.shape[0], x_offset:x_offset + cropped_detection.shape[1]] = cropped_detection

    # Combine the overlay with the original image
    b, g, r, a = cv2.split(overlay)
    alpha = a / 255.0

    for c in range(3):
        coco_image[:, :, c] = (1.0 - alpha) * coco_image[:, :, c] + alpha * overlay[:, :, c]

    # Adjust polygon labels for all detections
    for polygon in detection_polygons:
        adjusted_polygon = []
        for x_poly, y_poly in polygon:
            # Scale to original dimensions
            x_abs = x_poly * image.shape[1]
            y_abs = y_poly * image.shape[0]

            # Translate to combined bounding box
            x_abs -= x
            y_abs -= y

            # Apply initial scale factor (resizing)
            if scale_factor < 1.0:
                x_abs *= scale_factor
                y_abs *= scale_factor
            
            x_abs *= random_scale_factor
            y_abs *= random_scale_factor

            # Translate to new position
            x_abs += x_offset
            y_abs += y_offset

            # Normalize back to the new image dimensions
            x_norm = x_abs / coco_image.shape[1]
            y_norm = y_abs / coco_image.shape[0]

            # Ensure coordinates are within bounds
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            adjusted_polygon.append((x_norm, y_norm))
        adjusted_polygons.append(adjusted_polygon)

    return coco_image, adjusted_polygons

def augment_image(image, polygons, current_subfolder, class_ids, h, w, skip_augmentations, mirror_weights, crop_weights,
                  overlay_weights, rotate_weights, rotation_random_vs_90_weights, 
                  overlay_min_max_scale, maintain_aspect_ratio_weights, 
                  zoom_weights, zoom_in_vs_out_weights, zoom_padding, coco_image_folder):
    
    if current_subfolder not in skip_augmentations['Mirror']:
        mirror_choice = random.choices([True, False], weights=mirror_weights, k=1)[0]
        if mirror_choice:
            image = mirror_image(image)
            polygons = [mirror_polygon(polygon) for polygon in polygons]

    if current_subfolder not in skip_augmentations['Crop']:
        crop_choice = random.choices([True, False], weights=crop_weights, k=1)[0]
        if crop_choice and polygons:
            image, polygons, class_ids = crop_image_and_polygons(image, polygons, class_ids)
            maintain_aspect_ratio_choice = random.choices([True, False], weights=maintain_aspect_ratio_weights, k=1)[0]
            if maintain_aspect_ratio_choice:
                image, polygons = pad_image_and_adjust_polygons(image, polygons, (h, w))
    
    
    if current_subfolder not in skip_augmentations['Zoom']:
        zoom_choice = random.choices([True, False], weights=zoom_weights, k=1)[0]
        if zoom_choice and polygons:
            zoom_in = random.choices([True, False], weights=zoom_in_vs_out_weights, k=1)[0]
            if zoom_in:
                image, polygons = zoom_in_image_and_polygons(image, polygons, zoom_padding[0], zoom_padding[1])
            else:
                image, polygons = zoom_out_image_and_polygons(image, polygons, zoom_padding[2], zoom_padding[3])

    if current_subfolder not in skip_augmentations['Rotate']:
        rotate_choice = random.choices([True, False], weights=rotate_weights, k=1)[0]
        if rotate_choice:
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)
            rotation_degree = get_rotation_angle(rotation_random_vs_90_weights)
            image = rotate_image(image, rotation_degree)
            new_w, new_h = image.shape[1], image.shape[0]
            new_center = (new_w / 2, new_h / 2)
            polygons = [rotate_polygon(polygon, rotation_degree, center, new_center, (w, h), (new_w, new_h)) for polygon in polygons]

    if current_subfolder not in skip_augmentations['Overlay']:
        overlay_choice = random.choices([True, False], weights=overlay_weights, k=1)[0]
        if overlay_choice and polygons:
            #overlay_scale_choice = random.choices([True, False], weights=overlay_scale_weights, k=1)[0]
            coco_images = [os.path.join(coco_image_folder, f) for f in os.listdir(coco_image_folder) if os.path.isfile(os.path.join(coco_image_folder, f))]
            coco_image_path = random.choice(coco_images)
            coco_image = cv2.imread(coco_image_path)
            image, polygons = overlay_detections_on_coco(coco_image, image, polygons, overlay_min_max_scale[0], overlay_min_max_scale[1])
    
    
    formatted_polygons = [['{}'.format(class_id), *polygon] for class_id, polygon in zip(class_ids, polygons)]
    return image, formatted_polygons
