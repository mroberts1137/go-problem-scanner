"""
Functions for transforming Go board images.
"""
import cv2
import numpy as np
from typing import Dict, Tuple


def orient_board(board_image: np.ndarray, corners) -> np.ndarray:
    """Transform the board image to a standard orientation.
    
    Args:
        board_image: Image containing the Go board
        corners: Dictionary mapping corner names to coordinates
        
    Returns:
        Oriented board image with white background
    """
    pts_src = np.array([
        corners['top-left'],
        corners['top-right'],
        corners['bottom-right'],
        corners['bottom-left']
    ], dtype=np.float32)

    content_width = 500
    content_height = 500

    padding = 50  # px
    output_width = content_width + 2 * padding
    output_height = content_height + 2 * padding

    pts_dst = np.array([
        [padding, padding],  # top-left
        [padding + content_width, padding],  # top-right
        [padding + content_width, padding + content_height],  # bottom-right
        [padding, padding + content_height]  # bottom-left
    ], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Use warpPerspective with white border
    warped = cv2.warpPerspective(
        board_image, mat,
        (output_width, output_height),
        borderValue=(255, 255, 255)  # White background for binary images
    )

    # Optional: threshold again if small gray values appear
    _, warped_thresh = cv2.threshold(warped, 210, 255, cv2.THRESH_BINARY)

    return warped_thresh
