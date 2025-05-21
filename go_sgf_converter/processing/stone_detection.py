"""
Functions for detecting Go stones in board images.
"""
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from go_sgf_converter.utils import sgf_utils, draw_utils
from go_sgf_converter.utils.debugger import Debugger


def detect_stones_geometric(board_image: np.ndarray, image_grid):
    """Detect stones using geometric grid calculation.

    Args:
        board_image: Image containing Go board
        image_grid: Dict with keys = (row, col): board_intersection, values = [x, y]: image_coordinate

    Returns:
        stones: {'black': [], 'white': []}
    """
    # Detect stones at each intersection
    stones = {'black': [], 'white': []}

    intensity_bins = {}

    for coord, point in image_grid.items():
        stone_color = detect_stone_at_point(board_image, point, intensity_bins)
        if stone_color:
            sgf_coord = sgf_utils.convert_coord_to_sgf_coord(coord)
            stones[stone_color].append(sgf_coord)

    print(f"DEBUG: Detected stones - Black: {stones['black']}, White: {stones['white']}")

    debugger = Debugger.get_instance()
    if debugger:
        hist_img = draw_utils.generate_histogram_image(intensity_bins, title="Detected Intensities")
        debugger.save_debug_image(hist_img, "intensity_histogram.png")

    return stones


def detect_stone_at_point(image: np.ndarray, point: Tuple[int, int], intensity_bins) -> Union[str, None]:
    """Detect stone at specific intersection.
    
    Args:
        image: Board image
        point: Coordinates of intersection (x, y)
        
    Returns:
        'black', 'white', or None if no stone detected
    """
    x, y = point
    radius = 15  # Detection radius

    # Ensure we don't go out of bounds
    height, width = image.shape[:2]
    if x - radius < 0 or x + radius >= width or y - radius < 0 or y + radius >= height:
        return None

    # Extract region around intersection
    region = image[y - radius:y + radius, x - radius:x + radius]
    # gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Calculate average intensity
    avg_intensity = float(np.mean(region))

    intensity_bins[tuple(point)] = avg_intensity

    # Look for circular patterns
    circles = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=15, minRadius=8, maxRadius=18)

    # if circles is not None and len(circles[0]) > 0:
        # Found circular pattern, determine color based on intensity
    if avg_intensity < 25:  # Dark threshold for black stones
        return 'white'
    elif avg_intensity > 200:  # Light threshold for white stones
        return 'black'

    return None
