import cv2
from PIL import Image
import os


class Debugger:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.debug_count = 0

    def save_debug_image(self, image, filename):
        """Save image with step information for debugging"""
        self.debug_count += 1
        save_path = os.path.join(self.processed_dir, str(self.debug_count).zfill(2) + "_" + filename)
        if len(image.shape) == 3:
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(image_rgb).save(save_path)
        else:
            # Grayscale image
            Image.fromarray(image).save(save_path)