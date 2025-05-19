"""
Drawing Utility Functions
"""
import cv2
import numpy as np
from typing import Tuple, Dict, Optional

from go_sgf_converter.utils import sgf_utils


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
    cv2.line(img, corners['top-left'], corners['top-right'], (0, 255, 0), 3)
    cv2.line(img, corners['top-right'], corners['bottom-right'], (0, 255, 0), 3)
    cv2.line(img, corners['bottom-right'], corners['bottom-left'], (0, 255, 0), 3)
    cv2.line(img, corners['bottom-left'], corners['top-left'], (0, 255, 0), 3)

    return img


def draw_inner_outer_borders(board_image: np.ndarray, inner_border, outer_border) -> np.ndarray:
    # Draw board boundaries debug image
    img = board_image.copy()

    # Ensure proper shape for contours: (N, 1, 2)
    inner_contour = inner_border.reshape((-1, 1, 2)).astype(np.int32)
    outer_contour = outer_border.reshape((-1, 1, 2)).astype(np.int32)

    cv2.drawContours(img, [inner_contour], -1, (0, 255, 0), 3)
    cv2.drawContours(img, [outer_contour], -1, (0, 0, 255), 3)

    return img


def draw_grid_points(board_image, image_grid):
    img = board_image.copy()

    for (row, col), point in image_grid.items():
        x, y = int(point[0]), int(point[1])

        # Draw circle
        cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

        # Draw label
        label = f"({row},{col})"
        cv2.putText(img, label, (x - 5, y - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return img


def draw_stones_on_board(board_image, stones, grid_points):
    """
    Draws black and white stones onto the board image using the given coordinates.

    Parameters:
        board_image (np.ndarray): The base image of the board.
        stones (dict): {'black': [(row, col), ...], 'white': [(row, col), ...]}
        grid_points (dict): {(row, col): (x, y)} pixel coordinate mapping.

    Returns:
        np.ndarray: Image with stones drawn.
    """
    debug_image = board_image.copy()

    for color in ['black', 'white']:
        for sgf_coord in stones[color]:
            coord = sgf_utils.convert_sgf_coord_to_coord(sgf_coord)
            point = grid_points.get(coord)
            if point is None:
                continue  # skip invalid/missing coordinates

            # Define fill color and border color
            fill_color = (0, 0, 0) if color == 'black' else (255, 255, 255)
            border_color = (255, 0, 0)

            # Draw filled circle
            cv2.circle(debug_image, point, 15, fill_color, -1)
            # Draw border
            cv2.circle(debug_image, point, 15, border_color, 2)

            # Optional: Label the stone's grid position
            label = f"{coord[0]},{coord[1]}"
            cv2.putText(debug_image, label, (point[0] - 10, point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    return debug_image
