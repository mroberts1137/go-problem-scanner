"""
Image processing functions.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


def invert_image(board_image: np.ndarray):
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    return thresh


def thicken_lines(board_image: np.ndarray):
    # Define the kernel (controls how much to thicken)
    kernel = np.ones((3, 3), np.uint8)  # You can try (5, 5) for even thicker lines

    # Apply dilation
    thickened = cv2.dilate(board_image, kernel, iterations=1)

    return thickened
