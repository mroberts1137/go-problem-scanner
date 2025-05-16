import random
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import os
from typing import List, Tuple, Dict


def detect_lines(board_image: np.ndarray):
    """Find grid corner and spacing using line detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(board_image, cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=25)

    return edges, lines


def draw_board_lines(board_image, lines):
    # Create visualization of detected lines
    line_image = board_image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return line_image


def line_classifier(lines):
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


def detect_board_corners(h_lines: List, v_lines: List):
    """Detect which corner of the board is being shown"""

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

    # corner_pts = top_left, top_right, bottom_right, bottom_left

    corners = {
        'top-left': tuple(map(int, top_left)),
        'top-right': tuple(map(int, top_right)),
        'bottom-right': tuple(map(int, bottom_right)),
        'bottom-left': tuple(map(int, bottom_left))
    }

    return corners


def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, -c  # Return as Ax + By + C = 0


def intersection(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None  # Parallel lines
    x = -(b1 * c2 - b2 * c1) / det
    y = -(c1 * a2 - c2 * a1) / det
    return x, y


def draw_board_corners(board_image, corners):
    # Draw board boundaries debug image
    img = board_image.copy()
    cv2.line(img, corners['top-left'], corners['top-right'], (0, 255, 0), 2)
    cv2.line(img, corners['top-right'], corners['bottom-right'], (0, 255, 0), 2)
    cv2.line(img, corners['bottom-right'], corners['bottom-left'], (0, 255, 0), 2)
    cv2.line(img, corners['bottom-left'], corners['top-left'], (0, 255, 0), 2)

    return img
