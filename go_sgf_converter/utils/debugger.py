"""
Debugging utilities for visualizing processing steps.
"""
import cv2
from PIL import Image
import os


class Debugger:
    """Utility class for saving intermediate processing steps for debugging."""
    
    def __init__(self, processed_dir):
        """Initialize a debugger with a directory to save debug images.
        
        Args:
            processed_dir: Directory path to save debug images
        """
        self.processed_dir = processed_dir
        self.debug_count = 0
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)

    def save_debug_image(self, image, filename):
        """Save image with step information for debugging.
        
        Args:
            image: The image to save (numpy array)
            filename: Base filename for the debug image
        """
        self.debug_count += 1
        save_path = os.path.join(self.processed_dir, str(self.debug_count).zfill(2) + "_" + filename)
        
        if len(image.shape) == 3:
            # Convert BGR to RGB for PIL if needed (based on image source)
            try:
                # Try to detect if it's BGR from OpenCV
                if image[0, 0, 0] > 200 and image[0, 0, 2] < 50:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                Image.fromarray(image_rgb).save(save_path)
            except Exception:
                # Fall back to direct save if conversion fails
                Image.fromarray(image).save(save_path)
        else:
            # Grayscale image
            Image.fromarray(image).save(save_path)
