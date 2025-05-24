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
        cv2.imwrite("preprocess/1-blurred.jpg", blurred)

        # Apply adaptive thresholding for better text detection
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite("preprocess/2-binary.jpg", binary)

        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(binary, -1, kernel)
        cv2.imwrite("preprocess/3-sharpened.jpg", sharpened)

        # binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 2)
        # cv2.imwrite("preprocess/3-binary.jpg", binary)

        blurred_image = cv2.medianBlur(sharpened, 5)
        cv2.imwrite("preprocess/4-blurred_image.jpg", blurred_image)

        # _, binary = cv2.threshold(blurred_image3, 127, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("preprocess/7-blurred_image3.jpg", binary)

        # Invert image (white background -> black, black text -> white)
        # inverted = cv2.bitwise_not(binary)

        return blurred_image

    def find_problem_headers(self, image: np.ndarray) -> List[Dict]:
        """
        Find all "Problem ##" headers in the image using OCR with flexible matching.

        Args:
            image: Preprocessed image

        Returns:
            List of dictionaries containing problem info and bounding box info
        """
        # Use pytesseract to get detailed OCR data
        custom_config = r'--oem 3 --psm 6'

        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        problems = []

        # More flexible pattern to match "Problem" with common OCR errors
        # Handles: Problem, Preblem, Problen, problem, etc.
        flexible_patterns = [
            r'(.)?[Pp][rn][eoa][bhb][il][eoa][mn](.*)?',  # Main pattern with common substitutions
            r'(.)?[Pp]r[eo]bl[eoa][mn](.*)?',  # Simplified version
            r'(.)?[Pp][rn][eoa][bhb][1il][eoa][mn](.*)?',  # Include 1 for l
            r'(.)?[Pp][rRn][e3o0][bB][1iIlL][e3aA][mM](.*)?',  # General pattern
            r'(.)?[Pp]r[eo]bl[eao][mn](.*)?',  # Simpler fuzzy version
        ]

        # Check if text matches any of our flexible patterns
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            print(i, ': ', word)
            if not word:
                continue  # Skip empty or whitespace entries

            for pattern in flexible_patterns:
                if re.fullmatch(pattern, word, re.IGNORECASE):
                    bbox = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    center_x = data['left'][i] + data['width'][i] // 2
                    center_y = data['top'][i] + data['height'][i] // 2
                    problems.append({
                        'text': word,
                        'bbox': bbox,
                        'center_x': center_x,
                        'center_y': center_y,
                        'index': i
                    })
                    break

        return problems

    def estimate_problem_regions(self, problems: List[Dict], image_shape: Tuple[int, ...]) -> List[Dict]:
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

        # Estimate grid layout based on positions
        # Group problems by rows (similar y-coordinates)
        rows = []
        row_tolerance = max(25, height // 80)  # Adaptive tolerance based on image size

        for prob in problems:
            placed = False
            for row in rows:
                # Check if this problem belongs to an existing row
                row_center_y = sum(p['center_y'] for p in row) / len(row)
                if abs(prob['center_y'] - row_center_y) < row_tolerance:
                    row.append(prob)
                    placed = True
                    break
            if not placed:
                rows.append([prob])

        # Sort each row by x-coordinate
        for row in rows:
            row.sort(key=lambda p: p['center_x'])

        # Sort rows by y-coordinate
        rows.sort(key=lambda row: sum(p['center_y'] for p in row) / len(row))

        print(f"Detected grid: {len(rows)} rows with {[len(row) for row in rows]} problems each")

        regions = []
        padding = 80

        for row_idx, row in enumerate(rows):
            for col_idx, prob in enumerate(row):
                header_bbox = prob['bbox']

                # Estimate LEFT boundary
                if col_idx == 0:
                    left = 0
                else:
                    prev_prob = row[col_idx - 1]
                    left = (prev_prob['center_x'] + prob['center_x']) // 2 - padding

                # Estimate RIGHT boundary
                if col_idx == len(row) - 1:
                    right = width
                else:
                    next_prob = row[col_idx + 1]
                    right = (prob['center_x'] + next_prob['center_x']) // 2 + padding

                # Estimate TOP boundary
                if row_idx == 0:
                    top = 0
                else:
                    # prev_row_center = sum(p['center_y'] for p in rows[row_idx - 1]) / len(rows[row_idx - 1])
                    current_row_center = sum(p['center_y'] for p in row) / len(row)
                    top = int(current_row_center - 2 * padding)

                # Estimate BOTTOM boundary
                if row_idx == len(rows) - 1:
                    bottom = height
                else:
                    # current_row_center = sum(p['center_y'] for p in row) / len(row)
                    next_row_center = sum(p['center_y'] for p in rows[row_idx + 1]) / len(rows[row_idx + 1])
                    bottom = int(next_row_center - padding)

                # Ensure minimum dimensions
                # region_width = right - left
                # region_height = bottom - top

                # if region_width < self.min_problem_width:
                #     center_x = left + region_width // 2
                #     left = max(0, center_x - self.min_problem_width // 2)
                #     right = min(width, left + self.min_problem_width)
                #
                # if region_height < self.min_problem_height:
                #     center_y = top + region_height // 2
                #     top = max(0, center_y - self.min_problem_height // 2)
                #     bottom = min(height, top + self.min_problem_height)

                # Add padding
                padding_w = int((right - left) * self.padding_factor)
                padding_h = int((bottom - top) * self.padding_factor)

                left = max(0, left - padding_w)
                right = min(width, right + padding_w)
                top = max(0, top - padding_h)
                bottom = min(height, bottom + padding_h)

                regions.append({
                    # 'number': prob['number'],
                    # 'estimated_number': prob['estimated_number'],
                    'bbox': (left, top, right - left, bottom - top),
                    'header_bbox': header_bbox,
                    'header_text': prob['text'],
                    'row': row_idx,
                    'col': col_idx
                })

        return regions

    def extract_and_save_problems(self, original_image: np.ndarray, regions: List[Dict]) -> None:
        """
        Extract problem regions and save them as separate images.

        Args:
            original_image: Original input image
            regions: List of problem regions to extract
        """
        for idx, region in enumerate(regions):
            # problem_num = region['number']
            x, y, w, h = region['bbox']

            # Extract the region from the original image
            problem_image = original_image[y:y + h, x:x + w]

            # Save the image
            filename = f"problem_images/problem-{idx:03d}.jpg"
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
        print("Preprocessing complete.")

        if debug:
            cv2.imwrite("debug_preprocessed.jpg", preprocessed)

        # Find problem headers
        problems = self.find_problem_headers(preprocessed)
        print(f"Found {len(problems)} problems:")
        for prob in problems:
            print(f"  Problem: {prob['text']} at {prob['bbox']}")

        if not problems:
            print("No problems detected. Check image quality and OCR settings.")
            return

        if debug:
            draw_problem_words(original_image, problems, "debug/problem_text_detection.jpg")

        # Estimate problem regions
        # rows = self.estimate_problem_regions(problems, original_image.shape)
        #
        # if debug:
        #     draw_rows(original_image, rows, "debug/problem_rows.jpg")

        regions = self.estimate_problem_regions(problems, original_image.shape)

        # Create debug image showing detected regions
        if debug:
            draw_regions(original_image, regions, "debug/problem_region_detection.jpg")

        # Extract and save individual problems
        self.extract_and_save_problems(original_image, regions)

        print(f"Extraction complete! Saved {len(regions)} problems to problem_images/")


def draw_problem_words(image, data, filename):
    debug_image = image.copy()
    for word in data:
        x, y, w, h = word['bbox']
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 8)

        # Show both the detected text and assigned number
        cv2.putText(debug_image, word['text'], (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)
    cv2.imwrite(filename, debug_image)
    print(f"Debug images saved: {filename}")


def draw_regions(image, data, filename):
    debug_image = image.copy()
    for idx, region in enumerate(data):
        x, y, w, h = region['bbox']
        color = (255 * (idx % 2), 255 * (idx+1 % 2), 0)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 8)

        # Show both the detected text and assigned number
        cv2.putText(debug_image, region['header_text'], (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)
    cv2.imwrite(filename, debug_image)
    print(f"Debug images saved: {filename}")


def draw_rows(image, rows, filename):
    debug_image = image.copy()
    for row_idx, row in enumerate(rows):
        print(row_idx, row)
        row_center_y = int(round(sum(p['center_y'] for p in row) / len(row)))
        cv2.line(debug_image, (0, row_center_y), (image.shape[1], row_center_y), (255, 0, 0), 8)
    cv2.imwrite(filename, debug_image)
    print(f"Debug images saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Extract Go problems from scanned book pages')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    parser.add_argument('--padding', type=float, default=0,
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
