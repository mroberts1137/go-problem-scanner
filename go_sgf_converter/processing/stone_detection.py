"""
Functions for detecting Go stones in board images.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union

from go_sgf_converter.utils import sgf_utils


def detect_stone_at_point(image: np.ndarray, point: Tuple[int, int]) -> Union[str, None]:
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
    avg_intensity = np.mean(region)

    # Look for circular patterns
    circles = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=15, minRadius=8, maxRadius=18)
    print(f'check: {point}, {avg_intensity}')

    # if circles is not None and len(circles[0]) > 0:
        # Found circular pattern, determine color based on intensity
    if avg_intensity < 30:  # Dark threshold for black stones
        return 'white'
    elif avg_intensity > 200:  # Light threshold for white stones
        return 'black'

    return None


def find_grid_parameters(h_lines: List, v_lines: List) -> Dict:
    """Find grid corner and spacing using line detection.
    
    Args:
        h_lines: Horizontal lines
        v_lines: Vertical lines
        
    Returns:
        Dictionary containing 'h_spacing', 'v_spacing'
    """
    # Find board corner and spacing
    if not h_lines or not v_lines:
        raise ValueError("No lines specified.")

    # Extract coordinates and calculate spacing as before
    v_coords = []
    for line in v_lines:
        x1, y1, x2, y2 = line
        v_coords.append((x1 + x2) // 2)

    h_coords = []
    for line in h_lines:
        x1, y1, x2, y2 = line
        h_coords.append((y1 + y2) // 2)

    # Sort coordinates
    v_coords.sort()
    h_coords.sort()

    # Calculate spacing
    def find_most_common_spacing(coords):
        if len(coords) < 2:
            return 30  # Default spacing
        differences = []
        # line_diffs = []
        for i in range(len(coords) - 1):
            diff = coords[i + 1] - coords[i]
            # line_diffs.append(diff)
            if 10 <= diff <= 65:  # Reasonable range for grid spacing
                differences.append(diff)
        # print(line_diffs)
        if differences:
            return int(np.median(differences))
        return 30

    spacing_x = find_most_common_spacing(v_coords) + 4
    spacing_y = find_most_common_spacing(h_coords) + 4

    print(f"DEBUG: Calculated spacing - X: {spacing_x}, Y: {spacing_y}")

    return {'h_spacing': spacing_x, 'v_spacing': spacing_y}


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

    for coord, point in image_grid.items():
        stone_color = detect_stone_at_point(board_image, point)
        if stone_color:
            sgf_coord = sgf_utils.convert_coord_to_sgf_coord(coord)
            stones[stone_color].append(sgf_coord)

    print(f"DEBUG: Detected stones - Black: {stones['black']}, White: {stones['white']}")
    return stones

