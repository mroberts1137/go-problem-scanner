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
    # pts_src = np.array([
    #     corners['top-left'],
    #     corners['top-right'],
    #     corners['bottom-right'],
    #     corners['bottom-left']
    # ], dtype=np.float32)
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners, dtype=np.float32)
    else:
        corners = corners.astype(np.float32)

        # Sort the 4 corners into: top-left, top-right, bottom-right, bottom-left

    def sort_corners(pts: np.ndarray) -> np.ndarray:
        # Sort by y (top to bottom)
        pts = pts[np.argsort(pts[:, 1])]
        top_two = pts[:2]
        bottom_two = pts[2:]

        # Sort top two by x (left to right)
        top_left, top_right = top_two[np.argsort(top_two[:, 0])]
        # Sort bottom two by x (left to right)
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    pts_src = sort_corners(corners)

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
