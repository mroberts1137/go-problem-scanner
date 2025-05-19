"""
Functions for extracting features from Go board images.
"""
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from go_sgf_converter.processing import image_processing as ip


def get_board_border(board_image: np.ndarray, debugger):
    inverted_image = ip.invert_image(board_image)
    # debugger.save_debug_image(inverted_image, "pre-contour_thresh.jpg")

    # Apply dilation
    thickened = ip.thicken_lines(inverted_image)
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
    debugger.save_debug_image(contour_image, "contour_image.jpg")

    box_image = board_image.copy()
    cv2.drawContours(box_image, [box], 0, (0, 0, 255), 3)
    debugger.save_debug_image(box_image, "border_box_image.jpg")

    return box


def get_boundaries(border):
    top = (border[0, 1] + border[1, 1]) / 2
    bottom = (border[3, 1] + border[2, 1]) / 2
    left = (border[0, 0] + border[3, 0]) / 2
    right = (border[1, 0] + border[2, 0]) / 2
    return {'top': top, 'bottom': bottom, 'left': left, 'right': right}


def get_board_edges(outer_border, inner_border):
    edges = []

    outer_boundaries = get_boundaries(outer_border)
    inner_boundaries = get_boundaries(inner_border)

    differences = {}
    for side in ['top', 'bottom', 'left', 'right']:
        differences[side] = outer_boundaries[side] - inner_boundaries[side]
        if np.abs(differences[side]) < 10:
            edges.append(side)

    print(outer_boundaries)
    print(inner_boundaries)
    print(differences)

    return edges


def get_corner_points(corners, edges):
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


def construct_board_grid(corner_points, grid_spacing):
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


def detect_lines(board_image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Find grid lines in the board image.
    
    Args:
        board_image: Image containing Go board
        
    Returns:
        Tuple of (edges, lines) where edges is grayscale edge image and
        lines is array of line segments or None if no lines detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=25)

    return edges, lines


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
    return np.array([
        corners['top-left'],
        corners['top-right'],
        corners['bottom-right'],
        corners['bottom-left']
    ], dtype=np.float32)


def detect_board_corners(h_lines: List, v_lines: List) -> Dict[str, Tuple[int, int]]:
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
