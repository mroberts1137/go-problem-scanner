"""
Functions for transforming Go board images.
"""
import cv2
import numpy as np
from typing import Dict, Tuple

from go_sgf_converter.utils.debugger import Debugger


def orient_board(board_image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Transform the board image to a standard orientation with correct aspect ratio.

    Args:
        board_image: Image containing the Go board
        corners: Array of border corner coordinates

    Returns:
        Oriented board image with white background and correct aspect ratio
    """
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners, dtype=np.float32)
    else:
        corners = corners.astype(np.float32)

    # Calculate aspect ratio from the convex hull corners
    width1 = np.linalg.norm(corners[1] - corners[0])
    width2 = np.linalg.norm(corners[2] - corners[3])
    height1 = np.linalg.norm(corners[3] - corners[0])
    height2 = np.linalg.norm(corners[2] - corners[1])

    avg_width = (width1 + width2) / 2
    avg_height = (height1 + height2) / 2

    aspect_ratio = avg_height / avg_width
    print(f"Using convex hull aspect ratio: {aspect_ratio:.3f}")

    # Calculate output dimensions
    base_size = 500
    padding = 50

    if aspect_ratio <= 1:
        content_width = base_size
        content_height = int(base_size * aspect_ratio)
    else:
        content_width = int(base_size * aspect_ratio)
        content_height = base_size

    output_width = content_width + 2 * padding
    output_height = content_height + 2 * padding

    pts_dst = np.array([
        [padding, padding],
        [padding + content_width, padding],
        [padding + content_width, padding + content_height],
        [padding, padding + content_height]
    ], dtype=np.float32)

    mat = cv2.getPerspectiveTransform(corners, pts_dst)

    # Final warp with correct aspect ratio
    warped = cv2.warpPerspective(
        board_image, mat,
        (output_width, output_height),
        borderValue=(255, 255, 255)
    )

    # Optional: threshold again if small gray values appear
    _, warped_thresh = cv2.threshold(warped, 210, 255, cv2.THRESH_BINARY)

    debugger = Debugger.get_instance()
    if debugger:
        debugger.save_debug_image(warped_thresh, f"oriented_board_hull_aspect_{aspect_ratio:.2f}.jpg")

    return warped_thresh
