import cv2
import numpy as np
import pytesseract
import re
import os
from typing import List, Tuple, Dict
import argparse


class GoProblemParser:
    def __init__(self, padding_factor: float = 0.1, min_problem_height: int = 200, min_problem_width: int = 150):
        """
        Initialize the Go problem parser.

        Args:
            padding_factor: Factor to add padding around detected problems (0.1 = 10% padding)
            min_problem_height: Minimum height for a problem region
            min_problem_width: Minimum width for a problem region
        """
        self.padding_factor = padding_factor
        self.min_problem_height = min_problem_height
        self.min_problem_width = min_problem_width

        # Create output directory
        os.makedirs("problem_images", exist_ok=True)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better OCR and analysis.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Invert image (white background -> black, black text -> white)
        inverted = cv2.bitwise_not(binary)

        return inverted

    def find_problem_headers(self, image: np.ndarray) -> List[Dict]:
        """
        Find all "Problem ##" headers in the image using OCR.

        Args:
            image: Preprocessed image

        Returns:
            List of dictionaries containing problem number and bounding box info
        """
        # Use pytesseract to get detailed OCR data
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ProblemABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        problems = []
        problem_pattern = re.compile(r'Problem\s+(\d+)', re.IGNORECASE)

        # Group words that are close together to form potential problem headers
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text and data['conf'][i] > 30:  # Confidence threshold
                # Look for "Problem" followed by number in next words
                if 'problem' in text.lower():
                    # Try to find the complete problem header
                    header_text = text
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                    # Look ahead for the number part
                    for j in range(i + 1, min(i + 5, n_boxes)):
                        next_text = data['text'][j].strip()
                        if next_text and data['conf'][j] > 30:
                            # Check if next word is close enough (same line)
                            next_x = data['left'][j]
                            next_y = data['top'][j]
                            if abs(next_y - y) < h and next_x - (x + w) < 50:
                                header_text += " " + next_text
                                w = next_x + data['width'][j] - x  # Extend bounding box
                            else:
                                break

                    # Check if we found a valid problem header
                    match = problem_pattern.search(header_text)
                    if match:
                        problem_num = int(match.group(1))
                        problems.append({
                            'number': problem_num,
                            'text': header_text,
                            'bbox': (x, y, w, h),
                            'center_x': x + w // 2,
                            'center_y': y + h // 2
                        })

        # Sort problems by position (top to bottom, left to right)
        problems.sort(key=lambda p: (p['center_y'], p['center_x']))

        # Remove duplicates (same problem detected multiple times)
        unique_problems = []
        for prob in problems:
            is_duplicate = False
            for existing in unique_problems:
                if (abs(prob['center_x'] - existing['center_x']) < 100 and
                        abs(prob['center_y'] - existing['center_y']) < 50 and
                        prob['number'] == existing['number']):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_problems.append(prob)

        return unique_problems

    def estimate_problem_regions(self, problems: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Estimate the bounding regions for each problem based on header positions.

        Args:
            problems: List of detected problem headers
            image_shape: Shape of the image (height, width)

        Returns:
            List of problem regions with estimated bounding boxes
        """
        height, width = image_shape[:2]

        if not problems:
            return []

        # Estimate grid layout
        # Group problems by rows (similar y-coordinates)
        rows = []
        row_tolerance = 50

        for prob in problems:
            placed = False
            for row in rows:
                if abs(prob['center_y'] - row[0]['center_y']) < row_tolerance:
                    row.append(prob)
                    placed = True
                    break
            if not placed:
                rows.append([prob])

        # Sort each row by x-coordinate
        for row in rows:
            row.sort(key=lambda p: p['center_x'])

        # Sort rows by y-coordinate
        rows.sort(key=lambda row: row[0]['center_y'])

        regions = []

        for row_idx, row in enumerate(rows):
            for col_idx, prob in enumerate(row):
                # Estimate problem boundaries
                header_bbox = prob['bbox']
                header_x, header_y, header_w, header_h = header_bbox

                # Estimate left and right boundaries
                if col_idx == 0:
                    left = 0
                else:
                    prev_prob = row[col_idx - 1]
                    left = (prev_prob['center_x'] + prob['center_x']) // 2 - 20

                if col_idx == len(row) - 1:
                    right = width
                else:
                    next_prob = row[col_idx + 1]
                    right = (prob['center_x'] + next_prob['center_x']) // 2 + 20

                # Estimate top and bottom boundaries
                if row_idx == 0:
                    top = 0
                else:
                    prev_row = rows[row_idx - 1]
                    # Find the closest problem in the previous row
                    prev_y = max(p['center_y'] for p in prev_row)
                    top = (prev_y + prob['center_y']) // 2 - 30

                if row_idx == len(rows) - 1:
                    bottom = height
                else:
                    next_row = rows[row_idx + 1]
                    # Find the closest problem in the next row
                    next_y = min(p['center_y'] for p in next_row)
                    bottom = (prob['center_y'] + next_y) // 2 + 30

                # Ensure minimum dimensions
                region_width = right - left
                region_height = bottom - top

                if region_width < self.min_problem_width:
                    center_x = left + region_width // 2
                    left = max(0, center_x - self.min_problem_width // 2)
                    right = min(width, left + self.min_problem_width)

                if region_height < self.min_problem_height:
                    center_y = top + region_height // 2
                    top = max(0, center_y - self.min_problem_height // 2)
                    bottom = min(height, top + self.min_problem_height)

                # Add padding
                padding_w = int(region_width * self.padding_factor)
                padding_h = int(region_height * self.padding_factor)

                left = max(0, left - padding_w)
                right = min(width, right + padding_w)
                top = max(0, top - padding_h)
                bottom = min(height, bottom + padding_h)

                regions.append({
                    'number': prob['number'],
                    'bbox': (left, top, right - left, bottom - top),
                    'header_bbox': header_bbox
                })

        return regions

    def extract_and_save_problems(self, original_image: np.ndarray, regions: List[Dict]) -> None:
        """
        Extract problem regions and save them as separate images.

        Args:
            original_image: Original input image
            regions: List of problem regions to extract
        """
        for region in regions:
            problem_num = region['number']
            x, y, w, h = region['bbox']

            # Extract the region from the original image
            problem_image = original_image[y:y + h, x:x + w]

            # Save the image
            filename = f"problem_images/problem-{problem_num:03d}.jpg"
            cv2.imwrite(filename, problem_image)
            print(f"Saved {filename} (region: {x}, {y}, {w}, {h})")

    def process_page(self, image_path: str, debug: bool = False) -> None:
        """
        Process a complete go problem book page.

        Args:
            image_path: Path to the input image
            debug: Whether to save debug images
        """
        # Load the original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"Processing image: {image_path}")
        print(f"Image dimensions: {original_image.shape}")

        # Preprocess for OCR
        preprocessed = self.preprocess_image(original_image)

        if debug:
            cv2.imwrite("debug_preprocessed.jpg", preprocessed)

        # Find problem headers
        problems = self.find_problem_headers(preprocessed)
        print(f"Found {len(problems)} problems:")
        for prob in problems:
            print(f"  Problem {prob['number']}: {prob['text']} at {prob['bbox']}")

        if not problems:
            print("No problems detected. Check image quality and OCR settings.")
            return

        # Estimate problem regions
        regions = self.estimate_problem_regions(problems, original_image.shape)

        # Create debug image showing detected regions
        if debug:
            debug_image = original_image.copy()
            for region in regions:
                x, y, w, h = region['bbox']
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_image, f"P{region['number']}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite("debug_regions.jpg", debug_image)

        # Extract and save individual problems
        self.extract_and_save_problems(original_image, regions)

        print(f"Extraction complete! Saved {len(regions)} problems to problem_images/")


def main():
    parser = argparse.ArgumentParser(description='Extract Go problems from scanned book pages')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Padding factor around problems (default: 0.1)')
    parser.add_argument('--min-height', type=int, default=200,
                        help='Minimum problem height (default: 200)')
    parser.add_argument('--min-width', type=int, default=150,
                        help='Minimum problem width (default: 150)')

    args = parser.parse_args()

    # Create parser instance
    parser_instance = GoProblemParser(
        padding_factor=args.padding,
        min_problem_height=args.min_height,
        min_problem_width=args.min_width
    )

    try:
        # Process the page
        parser_instance.process_page(args.image_path, debug=args.debug)
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
