"""
Drawing Utility Functions
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def draw_board_lines(board_image: np.ndarray, lines: Optional[np.ndarray]) -> np.ndarray:
    """Create a visualization of detected board lines.

    Args:
        board_image: Original board image
        lines: Array of line segments from HoughLinesP

    Returns:
        Image with lines drawn
    """
    # Create visualization of detected lines
    line_image = board_image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return line_image


def draw_board_corners(board_image: np.ndarray, corners: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Draw board corners on image for debugging.

    Args:
        board_image: Original board image
        corners: Dictionary mapping corner names to coordinates

    Returns:
        Image with board corners drawn
    """
    # Draw board boundaries debug image
    img = board_image.copy()
    cv2.line(img, corners['top-left'], corners['top-right'], (0, 255, 0), 2)
    cv2.line(img, corners['top-right'], corners['bottom-right'], (0, 255, 0), 2)
    cv2.line(img, corners['bottom-right'], corners['bottom-left'], (0, 255, 0), 2)
    cv2.line(img, corners['bottom-left'], corners['top-left'], (0, 255, 0), 2)

    return img