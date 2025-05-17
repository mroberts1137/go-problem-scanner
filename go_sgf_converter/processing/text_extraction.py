"""
Functions for extracting text information from Go problem images.
"""
import cv2
import numpy as np
import pytesseract
import re
from typing import Tuple, Union


def extract_problem_number(problem_region: np.ndarray) -> Union[int, str]:
    """Extract problem number from the problem region.
    
    Args:
        problem_region: Image region containing problem title
        
    Returns:
        Extracted problem number as integer or empty string if not found
    """
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(problem_region, cv2.COLOR_RGB2GRAY)

    # Use OCR
    text = pytesseract.image_to_string(gray)
    print(f"DEBUG: Raw problem text: '{text}'")

    # Clean up text
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    cleaned_text = ' '.join(cleaned_text.split())

    problem_number = ''

    match = re.search(r'Problem\s+(\d+)', cleaned_text, flags=re.IGNORECASE)
    if match:
        problem_number = int(match.group(1))
        print(f"DEBUG: found problem number: '{problem_number}'")
    else:
        print("No match found.")

    return problem_number


def extract_description(description_region: np.ndarray) -> Tuple[str, str]:
    """Extract problem description and player to move.
    
    Args:
        description_region: Image region containing problem description
        
    Returns:
        Tuple of (description, player) where player is 'B' for Black or 'W' for White
    """
    # Convert to grayscale for OCR
    gray = cv2.cvtColor(description_region, cv2.COLOR_RGB2GRAY)

    # Use OCR
    text = pytesseract.image_to_string(gray)
    print(f"DEBUG: Raw description text: '{text}'")

    # Clean up text
    # Remove non-ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    cleaned_text = ' '.join(cleaned_text.split())

    # Remove turn information
    description = re.sub(r'(Black|White)\s+t(o|e)\s+play\.?', '', cleaned_text, flags=re.IGNORECASE)
    description = description.strip()

    player = 'B'

    # Look for turn indicators
    if re.search(r'Black\s+t(o|e)\s+play', cleaned_text, re.IGNORECASE):
        print("DEBUG: Found 'Black to play'")
        player = 'B'
    elif re.search(r'White\s+t(o|e)\s+play', cleaned_text, re.IGNORECASE):
        print("DEBUG: Found 'White to play'")
        player = 'W'

    print(f"DEBUG: Cleaned description: '{description}'")
    return description, player
