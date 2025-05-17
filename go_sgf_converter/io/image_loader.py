"""
Utilities for loading and saving images.
"""
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in RGB format
    """
    print('load_image...')
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def save_sgf(sgf_content: str, filename: str):
    """Save SGF content to a file.
    
    Args:
        sgf_content: SGF formatted string
        filename: Path to save the file
    """
    with open(filename, 'w') as f:
        f.write(sgf_content)
    print(f"Saved SGF file: {filename}")
