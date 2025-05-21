"""
Functions for detecting Go stones in board images.
"""
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from go_sgf_converter.utils import sgf_utils, draw_utils
from go_sgf_converter.utils.debugger import Debugger


def detect_stones_geometric(board_image: np.ndarray, image_grid, grid_spacing):
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
        stone_color = detect_stone_at_point(board_image, point, grid_spacing, intensity_bins)
        if stone_color:
            if stone_color == 'black' or stone_color == 'white':
                sgf_coord = sgf_utils.convert_coord_to_sgf_coord(coord)
            stones[stone_color].append(sgf_coord)

    print(f"DEBUG: Detected stones - Black: {stones['black']}, White: {stones['white']}")

    debugger = Debugger.get_instance()
    if debugger:
        hist_img = draw_utils.generate_histogram_image(intensity_bins, title="Detected Intensities")
        debugger.save_debug_image(hist_img, "intensity_histogram.png")

    return stones


def detect_stone_at_point(image: np.ndarray, point: Tuple[int, int], grid_spacing, intensity_bins) -> Union[str, None]:
    """Detect stone at specific intersection.
    
    Args:
        image: Board image
        point: Coordinates of intersection (x, y)
        
    Returns:
        'black', 'white', or None if no stone detected
    """
    x, y = point
    h_spacing = grid_spacing['h_spacing']
    v_spacing = grid_spacing['v_spacing']
    max_radius = max(h_spacing, v_spacing) // 2 + 8
    radius = 15  # Detection radius

    # Ensure we don't go out of bounds
    height, width = image.shape[:2]
    if x - max_radius < 0 or x + max_radius >= width or y - max_radius < 0 or y + max_radius >= height:
        return None

    # Extract region around intersection
    avg_intensity_region = image[y - radius:y + radius, x - radius:x + radius]
    hough_region = image[y - max_radius-5:y + max_radius+5, x - max_radius-5:x + max_radius+5]

    # Calculate average intensity
    avg_intensity = float(np.mean(avg_intensity_region))

    intensity_bins[tuple(point)] = avg_intensity

    blurred_image = cv2.GaussianBlur(hough_region, (9, 9), 0)

    # Look for circular patterns
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=min(h_spacing, v_spacing)//2,
                               param1=50, param2=15, minRadius=max_radius//2, maxRadius=max_radius)

    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.round(circles[0, :]).astype("int")
        print(point, avg_intensity, circles)

        # Loop over the detected circles and draw them
        for (x, y, r) in circles:
            cv2.circle(blurred_image, (x, y), r, (255, 255, 0), 2)

        # TODO: Stitch all images together into a grid of all images in one image
        # Display the image with detected circles
        # cv2.imshow('Detected Circles', blurred_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # if circles is not None and len(circles[0]) > 0:
            # Found circular pattern, determine color based on intensity
        if avg_intensity < 50:  # Dark threshold for black stones
            return 'white'
        elif avg_intensity > 200:  # Light threshold for white stones
            return 'black'
        else:
            print('Anomaly at ', point, avg_intensity)

    return None
