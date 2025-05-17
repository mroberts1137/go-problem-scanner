"""
Functions for segmenting Go problem images into components.
"""
import cv2
import numpy as np
import pytesseract
import re
from typing import Tuple


def find_board_bounds_by_text(image: np.ndarray, debugger) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find board boundaries by locating Problem title and description text
    
    Args:
        image: Input image containing Go problem
        debugger: Debugger instance for saving intermediate images
        
    Returns:
        Tuple of (problem_region, board_region, description_region) as numpy arrays
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Get all text with positions
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    problem_y = None
    description_y = None

    # Create debug image showing detected text
    debug_image = image.copy()

    for i, text in enumerate(text_data['text']):
        if text.strip():
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            # Draw bounding box around text
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_image, text[:10], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Look for "Problem" text
            if re.search(r'Problem', text, re.IGNORECASE):
                problem_y = y + h  # Bottom of problem text
                print(f"DEBUG: Found 'Problem' text at y={problem_y}")

            # Look for common description words that indicate start of description
            if re.search(r'(to\s+play|Black|White)', text, re.IGNORECASE) and description_y is None:
                description_y = y  # Top of description text
                print(f"DEBUG: Found description text starting at y={description_y}")

    debugger.save_debug_image(debug_image, "text_detection.jpg")

    # If we couldn't find text bounds, use defaults
    if problem_y is None:
        problem_y = int(image.shape[0] * 0.1)  # 10% from top
        print(f"DEBUG: Using default problem_y={problem_y}")

    if description_y is None:
        description_y = int(image.shape[0] * 0.75)  # 75% from top
        print(f"DEBUG: Using default description_y={description_y}")

    # Add extra padding to avoid including text
    board_top = problem_y + 20  # Increased padding
    board_bottom = description_y - 5  # Increased padding

    board_bounds = {
        'top': board_top,
        'bottom': board_bottom,
        'left': 0,
        'right': image.shape[1]
    }

    # Extract board region
    problem_region = image[:board_bounds['top'], board_bounds['left']:board_bounds['right']]
    board_region = image[board_bounds['top']:board_bounds['bottom'], board_bounds['left']:board_bounds['right']]
    description_region = image[board_bounds['bottom']:, board_bounds['left']:board_bounds['right']]

    debugger.save_debug_image(problem_region, "problem_region.jpg")
    debugger.save_debug_image(description_region, "description_region.jpg")
    debugger.save_debug_image(board_region, "extracted_board.jpg")

    return problem_region, board_region, description_region
