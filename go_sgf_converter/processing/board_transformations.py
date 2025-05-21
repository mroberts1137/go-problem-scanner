"""
Functions for transforming Go board images.
"""
import cv2
import numpy as np
from typing import Dict, Tuple

from go_sgf_converter.utils.debugger import Debugger


def count_grid_lines(board_image: np.ndarray) -> tuple[int, int]:
    """Count the number of grid lines in the Go board image.

    Args:
        board_image: Binary image of the Go board

    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    # Ensure binary image
    if len(board_image.shape) > 2:
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = board_image.copy()

    # Convert to binary, handling both black-on-white and white-on-black
    mean_value = np.mean(gray)
    if mean_value > 127:  # white background
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:  # black background
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    debugger = Debugger.get_instance()
    if debugger:
        debugger.save_debug_image(binary, "grid_binary.jpg")

    # Remove stones using morphological operations
    # Stones are roughly circular and larger than line thickness
    kernel_size = max(3, min(binary.shape) // 50)  # Adaptive kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if debugger:
        debugger.save_debug_image(cleaned, "grid_cleaned.jpg")

    # Process horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1] // 3, 1))
    horizontal_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, horizontal_kernel)

    if debugger:
        debugger.save_debug_image(horizontal_lines, "horizontal_lines.jpg")

    # Process vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, binary.shape[0] // 3))
    vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, vertical_kernel)

    if debugger:
        debugger.save_debug_image(vertical_lines, "vertical_lines.jpg")

    # Project and count lines
    h_projection = np.sum(horizontal_lines, axis=1)
    v_projection = np.sum(vertical_lines, axis=0)

    def count_peaks(projection):
        # Normalize projection
        if np.max(projection) > 0:
            projection = projection / np.max(projection)

        # Use adaptive threshold based on the data
        threshold = np.mean(projection[projection > 0]) * 0.5

        # Find peaks
        peaks = []
        in_peak = False
        min_peak_distance = len(projection) // 25  # Minimum distance between peaks
        last_peak = -min_peak_distance

        for i, val in enumerate(projection):
            if val > threshold and not in_peak and (i - last_peak) >= min_peak_distance:
                peaks.append(i)
                in_peak = True
                last_peak = i
            elif val <= threshold:
                in_peak = False

        return len(peaks)

    h_count = count_peaks(h_projection)
    v_count = count_peaks(v_projection)

    return h_count, v_count


def orient_board(
        board_image: np.ndarray,
        corners: np.ndarray,
        use_hull_ratio: bool = True
) -> np.ndarray:
    """Transform the board image to a standard orientation with correct aspect ratio.

    Args:
        board_image: Image containing the Go board
        corners: Array of border corner coordinates
        use_hull_ratio: If True, use the convex hull aspect ratio instead of counting lines

    Returns:
        Oriented board image with white background and correct aspect ratio
    """
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners, dtype=np.float32)
    else:
        corners = corners.astype(np.float32)

    # Calculate hull aspect ratio
    hull_width = np.linalg.norm(corners[1] - corners[0])  # top edge
    hull_height = np.linalg.norm(corners[3] - corners[0])  # left edge
    hull_ratio = hull_width / hull_height
    print(hull_width, hull_height, hull_ratio)

    if not use_hull_ratio:
        # First, do a preliminary warp to get a roughly oriented board
        temp_size = 800
        temp_pts = np.array([
            [0, 0],
            [temp_size, 0],
            [temp_size, temp_size],
            [0, temp_size]
        ], dtype=np.float32)

        temp_mat = cv2.getPerspectiveTransform(corners, temp_pts)
        temp_warped = cv2.warpPerspective(
            board_image, temp_mat,
            (temp_size, temp_size),
            borderValue=(255, 255, 255)
        )

        # Count grid lines in the temporary warped image
        h_lines, v_lines = count_grid_lines(temp_warped)

        # Validate line counts
        valid_sizes = [9, 13, 19]
        if h_lines < 5 or v_lines < 5:
            print(f"Warning: Detected {h_lines}x{v_lines} lines. Falling back to hull ratio.")
            aspect_ratio = hull_ratio
        else:
            aspect_ratio = v_lines / h_lines
    else:
        aspect_ratio = hull_ratio

    # Determine output dimensions
    base_size = 500
    padding = 50

    if aspect_ratio <= 1:
        content_width = base_size
        content_height = int(base_size * aspect_ratio)
    else:
        content_width = int(base_size * aspect_ratio)
        content_height = base_size
    print(content_width, content_height)

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
        if use_hull_ratio:
            debugger.save_debug_image(warped_thresh, f"oriented_board_hull_ratio_{aspect_ratio:.2f}.jpg")
        else:
            debugger.save_debug_image(warped_thresh, f"oriented_board_grid_{h_lines}x{v_lines}.jpg")

    return warped_thresh


# def orient_board(board_image: np.ndarray, corners: np.ndarray) -> np.ndarray:
#     """Transform the board image to a standard orientation.
#
#     Args:
#         board_image: Image containing the Go board
#         corners: Array of border corner coordinates
#
#     Returns:
#         Oriented board image with white background
#     """
#     if not isinstance(corners, np.ndarray):
#         corners = np.array(corners, dtype=np.float32)
#     else:
#         corners = corners.astype(np.float32)
#
#     content_width = 500
#     content_height = 500
#
#     padding = 50  # px
#     output_width = content_width + 2 * padding
#     output_height = content_height + 2 * padding
#
#     pts_dst = np.array([
#         [padding, padding],  # top-left
#         [padding + content_width, padding],  # top-right
#         [padding + content_width, padding + content_height],  # bottom-right
#         [padding, padding + content_height]  # bottom-left
#     ], dtype=np.float32)
#
#     mat = cv2.getPerspectiveTransform(corners, pts_dst)
#
#     # Use warpPerspective with white border
#     warped = cv2.warpPerspective(
#         board_image, mat,
#         (output_width, output_height),
#         borderValue=(255, 255, 255)  # White background for binary images
#     )
#
#     # Optional: threshold again if small gray values appear
#     _, warped_thresh = cv2.threshold(warped, 210, 255, cv2.THRESH_BINARY)
#
#     return warped_thresh
