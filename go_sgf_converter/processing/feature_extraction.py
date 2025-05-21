"""
Functions for extracting features from Go board images.
"""
from typing import Dict

import numpy as np

from go_sgf_converter.utils.debugger import Debugger

# Import all geometric operations functions to maintain backward compatibility
from go_sgf_converter.processing.board_analyzer import (
    get_boundaries, get_board_edges, get_corner_points, construct_board_grid,
    get_image_grid, detect_lines, line_classifier, get_bounding_lines,
    corners_to_array, detect_board_corners, line_from_points, intersection,
    get_board_border
)


def extract_board_features(board_image: np.ndarray) -> Dict:
    """Extract features from a Go board image.

    Args:
        board_image: Input image of Go board

    Returns:
        Dictionary containing extracted board features
    """
    # GET BOARD IMAGE OUTER BORDER
    # Extract board border convex hull. This includes lines which extend beyond edge of image
    convex_hull_border = get_board_border(board_image)

    # GET BOARD IMAGE INNER BORDER
    # Detect lines in the image
    lines = detect_lines(board_image)

    # Classify lines
    h_lines, v_lines = line_classifier(lines)

    # Detect board corners
    corners = detect_board_corners(h_lines, v_lines)

    # Convert corners to array format. This does not include lines which extend beyond edge of image
    lines_border = corners_to_array(corners)

    # DETERMINE BOARD EDGES BY COMPARING OUTER AND INNER BORDERS
    # Get board edges
    board_edges = get_board_edges(convex_hull_border, lines_border)

    # Get corner points based on detected edges
    corner_points = get_corner_points(corners, board_edges)

    # Return extracted features
    return {
        'border_box': convex_hull_border,
        'lines': lines,
        'h_lines': h_lines,
        'v_lines': v_lines,
        'corners': corners,
        'lines_border': lines_border,
        'board_edges': board_edges,
        'corner_points': corner_points
    }


def build_board_grid(feature_data: Dict, board_image: np.ndarray) -> Dict:
    """Build a grid representation of the Go board.

    Args:
        feature_data: Dictionary of extracted board features
        board_image: Image of the board

    Returns:
        Dictionary containing grid data and image grid
    """
    # Find grid spacing from h_lines and v_lines
    from go_sgf_converter.processing.board_analyzer import find_grid_spacing
    grid_spacing = find_grid_spacing(feature_data['h_lines'], feature_data['v_lines'])

    # Construct board grid
    board_grid = construct_board_grid(feature_data['corner_points'], grid_spacing)

    # Filter grid points
    image_grid = get_image_grid(board_grid, feature_data['border_box'])

    return {
        'grid_spacing': grid_spacing,
        'board_grid': board_grid,
        'image_grid': image_grid
    }
