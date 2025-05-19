"""
Functions for extracting features from Go board images.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


def get_board_border(board_image: np.ndarray, debugger):
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, 0)
    debugger.save_debug_image(thresh, "pre-contour_thresh.jpg")

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found.")

    height, width = board_image.shape[:2]
    margin = 10  # pixel margin to define "too close to edge"

    def is_contour_touching_edge(contour):
        for point in contour:
            x, y = point[0]
            if x <= margin or x >= width - margin or y <= margin or y >= height - margin:
                return True
        return False

    # Filter out contours that touch the edge
    internal_contours = [cnt for cnt in contours if not is_contour_touching_edge(cnt)]

    if not internal_contours:
        raise ValueError("No internal contours found.")

    # Combine all internal contours into one big set of points
    all_points = np.vstack(internal_contours)

    # Get the convex hull of all points
    hull = cv2.convexHull(all_points)

    # Fit a rotated rectangle around the hull
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    contour_image = board_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 3)
    debugger.save_debug_image(contour_image, "contour_image.jpg")

    internal_contours_image = board_image.copy()
    cv2.drawContours(internal_contours_image, internal_contours, -1, (0, 255, 0), 2)
    debugger.save_debug_image(internal_contours_image, "internal_contours_image.jpg")

    box_image = board_image.copy()
    cv2.drawContours(box_image, [box], 0, (0, 255, 0), 2)
    debugger.save_debug_image(box_image, "border_box_image.jpg")

    return box

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
    cv2.line(img, corners['top-left'], corners['top-right'], (0, 255, 0), 2)
    cv2.line(img, corners['top-right'], corners['bottom-right'], (0, 255, 0), 2)
    cv2.line(img, corners['bottom-right'], corners['bottom-left'], (0, 255, 0), 2)
    cv2.line(img, corners['bottom-left'], corners['top-left'], (0, 255, 0), 2)

    return img
