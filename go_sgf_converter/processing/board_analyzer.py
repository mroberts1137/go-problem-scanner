"""
Functions for geometric operations related to Go board analysis.
This module contains functions that handle board geometry, line detection,
corner calculations and other geometric transformations.
"""
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from go_sgf_converter.utils.debugger import Debugger
from go_sgf_converter.utils import draw_utils


def get_boundaries(border):
    """Extract boundaries from a border array.

    Args:
        border: Array representing a border with points

    Returns:
        Dictionary containing top, bottom, left and right boundary coordinates
    """
    top = (border[0, 1] + border[1, 1]) / 2
    bottom = (border[3, 1] + border[2, 1]) / 2
    left = (border[0, 0] + border[3, 0]) / 2
    right = (border[1, 0] + border[2, 0]) / 2
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


def get_board_edges(outer_border, inner_border):
    """Determine which edges of the board are actual board edges.

    Args:
        outer_border: Outer border coordinates
        inner_border: Inner border coordinates from line detection

    Returns:
        List of edge names ('top', 'bottom', 'left', 'right') that are board edges
    """
    edges = []

    outer_boundaries = get_boundaries(outer_border)
    inner_boundaries = get_boundaries(inner_border)

    differences = {}
    for side in ['top', 'bottom', 'left', 'right']:
        differences[side] = outer_boundaries[side] - inner_boundaries[side]
        if np.abs(differences[side]) < 10:
            edges.append(side)

    print('edge differences: ', differences)

    return edges


def get_corner_points(corners, edges):
    """Extract corner points based on detected edges.

    Args:
        corners: Dictionary containing corner coordinates
        edges: List of detected edge names

    Returns:
        Dictionary of corner points that are part of detected edges
    """
    corner_points = {}
    if 'top' in edges and 'left' in edges:
        corner_points['top-left'] = corners['top-left']
    if 'top' in edges and 'right' in edges:
        corner_points['top-right'] = corners['top-right']
    if 'bottom' in edges and 'right' in edges:
        corner_points['bottom_right'] = corners['bottom-right']
    if 'bottom' in edges and 'left' in edges:
        corner_points['bottom-left'] = corners['bottom-left']

    return corner_points


def find_grid_spacing(h_lines: List, v_lines: List) -> Dict:
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

    debugger = Debugger.get_instance()
    if debugger:
        v_coords_hist_img = draw_utils.generate_histogram_image(v_coords, title="v_coords", bins=len(v_coords))
        debugger.save_debug_image(v_coords_hist_img, "v_coords.png")
        h_coords_hist_img = draw_utils.generate_histogram_image(h_coords, title="h_coords", bins=len(h_coords))
        debugger.save_debug_image(h_coords_hist_img, "h_coords.png")

    # Calculate spacing
    def find_most_common_spacing(coords):
        if len(coords) < 2:
            return 30  # Default spacing
        differences = []
        line_diffs = []
        for i in range(len(coords) - 1):
            diff = coords[i + 1] - coords[i]
            line_diffs.append(diff)
            if 10 <= diff <= 80:  # Reasonable range for grid spacing
                differences.append(diff)

        if debugger:
            diff_hist_img = draw_utils.generate_histogram_image(line_diffs, title="line_diffs")
            debugger.save_debug_image(diff_hist_img, "line_diffs.png")

        if differences:
            return int(np.median(differences))
        return 30

    spacing_x = find_most_common_spacing(v_coords) + 4
    spacing_y = find_most_common_spacing(h_coords) + 4

    print(f"DEBUG: Calculated spacing - X: {spacing_x}, Y: {spacing_y}")

    return {'h_spacing': spacing_x, 'v_spacing': spacing_y}


def construct_board_grid(corner_points, grid_spacing):
    """Construct a grid of points representing Go board intersections.

    Args:
        corner_points: Dictionary containing coordinates of board corners
        grid_spacing: Dictionary with horizontal and vertical grid spacing

    Returns:
        Dictionary mapping (row, col) coordinates to point coordinates
    """
    board_grid = {}

    # Extract top-left corner based on corner_points
    if 'top-left' in corner_points:
        top_left = np.array(corner_points['top-left'], dtype=np.int32)
    elif 'top-right' in corner_points:
        top_right = np.array(corner_points['top-right'], dtype=np.int32)
        top_left = top_right - np.array([18 * grid_spacing['h_spacing'], 0], dtype=np.int32)
    elif 'bottom-left' in corner_points:
        bottom_left = np.array(corner_points['bottom-left'], dtype=np.int32)
        top_left = bottom_left - np.array([0, 18 * grid_spacing['v_spacing']], dtype=np.int32)
    elif 'bottom-right' in corner_points:
        bottom_right = np.array(corner_points['bottom-right'], dtype=np.int32)
        top_left = bottom_right - np.array([18 * grid_spacing['h_spacing'], 18 * grid_spacing['v_spacing']], dtype=np.int32)
    else:
        raise ValueError("No corner points found")
    print('top-left: ', top_left)

    # Construct 2D board grid as a dict with (row, col) keys
    for row in range(19):
        for col in range(19):
            point = top_left + np.array([
                col * grid_spacing['h_spacing'],
                row * grid_spacing['v_spacing']
            ], dtype=np.int32)
            board_grid[(row, col)] = point

    return board_grid


def get_image_grid(board_grid, convex_hull_border):
    """Filter grid points to only include those within the board boundaries.

    Args:
        board_grid: Dictionary mapping (row, col) to point coordinates
        convex_hull_border: Array representing the boundary of the board

    Returns:
        Filtered dictionary with only points within the board boundaries
    """
    boundaries = get_boundaries(convex_hull_border)
    padding = 10

    # Apply padding to boundaries
    boundaries['left'] -= padding
    boundaries['right'] += padding
    boundaries['top'] -= padding
    boundaries['bottom'] += padding

    # Create a filtered grid dictionary
    image_grid = {}

    for row in range(19):
        for col in range(19):
            point = board_grid.get((row, col))
            if point is None:
                continue  # skip missing points

            x, y = point
            if boundaries['left'] < x < boundaries['right'] and boundaries['top'] < y < boundaries['bottom']:
                image_grid[(row, col)] = point

    return image_grid


def detect_lines(board_image: np.ndarray) -> Optional[np.ndarray]:
    """Find grid lines in the board image.

    Args:
        board_image: Image containing Go board

    Returns:
        lines: Array of line segments or None if no lines detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=25)

    return lines


def line_classifier(lines: Optional[np.ndarray]) -> Tuple[List[List[int]], List[List[int]]]:
    """Classify lines as horizontal or vertical.

    Args:
        lines: Array of line segments from HoughLinesP

    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Classify as horizontal or vertical
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < np.pi / 16 or angle > 15 * np.pi / 16:  # Horizontal
                horizontal_lines.append(line[0])
            elif 7 * np.pi / 16 < angle < 9 * np.pi / 16:  # Vertical
                vertical_lines.append(line[0])

    print(f"DEBUG: Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
    return horizontal_lines, vertical_lines


def get_bounding_lines(h_lines: List, v_lines: List) -> dict[str, tuple[float, float, float]]:
    """Get bounding lines for the Go board.

    Args:
        h_lines: List of horizontal line segments
        v_lines: List of vertical line segments

    Returns:
        Dictionary mapping edge names to line equations
    """
    # Find extreme line positions
    top_line = min(h_lines, key=lambda line: min(line[1], line[3]))
    bottom_line = max(h_lines, key=lambda line: max(line[1], line[3]))
    left_line = min(v_lines, key=lambda line: min(line[0], line[2]))
    right_line = max(v_lines, key=lambda line: max(line[0], line[2]))

    # Get coordinates of corners
    # Convert to line equations
    top_eq = line_from_points(top_line[:2], top_line[2:])
    bottom_eq = line_from_points(bottom_line[:2], bottom_line[2:])
    left_eq = line_from_points(left_line[:2], left_line[2:])
    right_eq = line_from_points(right_line[:2], right_line[2:])

    bounding_lines = {
        'top': top_eq,
        'bottom': bottom_eq,
        'left': left_eq,
        'right': right_eq
    }

    return bounding_lines


def corners_to_array(corners: dict) -> np.ndarray:
    """Convert corner dictionary to numpy array.

    Args:
        corners: Dictionary mapping corner names to coordinates

    Returns:
        Numpy array of corner coordinates
    """
    return np.array([
        corners['top-left'],
        corners['top-right'],
        corners['bottom-right'],
        corners['bottom-left']
    ], dtype=np.float32)


def detect_board_corners(h_lines: List, v_lines: List) -> Dict[str, tuple[int, int]]:
    """Detect corners of the Go board using line intersections.

    Args:
        h_lines: List of horizontal line segments
        v_lines: List of vertical line segments

    Returns:
        Dictionary mapping corner names ('top-left', etc.) to (x,y) coordinates
    """
    # Find extreme line positions
    top_line = min(h_lines, key=lambda line: min(line[1], line[3]))
    bottom_line = max(h_lines, key=lambda line: max(line[1], line[3]))
    left_line = min(v_lines, key=lambda line: min(line[0], line[2]))
    right_line = max(v_lines, key=lambda line: max(line[0], line[2]))

    # Get coordinates of corners
    # Convert to line equations
    top_eq = line_from_points(top_line[:2], top_line[2:])
    bottom_eq = line_from_points(bottom_line[:2], bottom_line[2:])
    left_eq = line_from_points(left_line[:2], left_line[2:])
    right_eq = line_from_points(right_line[:2], right_line[2:])

    # Compute intersections
    top_left = intersection(top_eq, left_eq)
    top_right = intersection(top_eq, right_eq)
    bottom_left = intersection(bottom_eq, left_eq)
    bottom_right = intersection(bottom_eq, right_eq)

    corners = {
        'top-left': tuple(map(int, top_left)),
        'top-right': tuple(map(int, top_right)),
        'bottom-right': tuple(map(int, bottom_right)),
        'bottom-left': tuple(map(int, bottom_left))
    }

    return corners


def line_from_points(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[float, float, float]:
    """Convert two points to line equation (Ax + By + C = 0).

    Args:
        p1: First point coordinates (x, y)
        p2: Second point coordinates (x, y)

    Returns:
        Coefficients of line equation (A, B, C)
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, -c  # Return as Ax + By + C = 0


def intersection(l1: Tuple[float, float, float], l2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    """Find intersection of two lines.

    Args:
        l1: First line coefficients (A, B, C)
        l2: Second line coefficients (A, B, C)

    Returns:
        Intersection point (x, y) or None if lines are parallel
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None  # Parallel lines
    x = -(b1 * c2 - b2 * c1) / det
    y = -(c1 * a2 - c2 * a1) / det
    return x, y


def get_board_border(board_image: np.ndarray):
    """Get the border of a Go board.

    Args:
        board_image: Image of the Go board

    Returns:
        Array representing the board border
    """
    from go_sgf_converter.processing import image_processing as ip

    inverted_image = ip.invert_image(board_image)

    # Apply dilation
    thickened = ip.thicken_lines(inverted_image)

    debugger = Debugger.get_instance()

    if debugger:
        debugger.save_debug_image(thickened, "pre-contour_thickened.jpg")

    contours, _ = cv2.findContours(thickened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found.")

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get min area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)

    def sort_corners(pts: np.ndarray) -> np.ndarray:
        # Sort by y (top to bottom)
        pts = pts[np.argsort(pts[:, 1])]
        top_two = pts[:2]
        bottom_two = pts[2:]

        # Sort top two by x (left to right)
        top_left, top_right = top_two[np.argsort(top_two[:, 0])]
        # Sort bottom two by x (left to right)
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    box = sort_corners(box)

    contour_image = board_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

    if debugger:
        debugger.save_debug_image(contour_image, "contour_image.jpg")

    box_image = board_image.copy()
    cv2.drawContours(box_image, [box], 0, (0, 0, 255), 3)

    if debugger:
        debugger.save_debug_image(box_image, "border_box_image.jpg")

    return box
