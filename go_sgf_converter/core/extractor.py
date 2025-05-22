import cv2
import numpy as np
import pytesseract
import re
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class GoProblemExtractor:
    def __init__(self, padding: int = 20, debug: bool = True):
        """
        Initialize the Go problem extractor.

        Args:
            padding: Pixels to add around each detected problem region
            debug: Whether to show intermediate processing steps
        """
        self.padding = padding
        self.debug = debug

        # Create output directory
        os.makedirs("problem_images", exist_ok=True)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image with thresholding and sharpening.

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

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(thresh, -1, kernel)

        # Invert image (white background, black text -> black background, white text)
        inverted = cv2.bitwise_not(sharpened)
        cv2.imwrite("preprocess/a-out.jpg", sharpened)

        blurred_image = cv2.medianBlur(sharpened, 5)
        cv2.imwrite("preprocess/b-out.jpg", blurred_image)

        # if self.debug:
        #     self._show_debug_images([
        #         (gray, "Original Grayscale"),
        #         (thresh, "Thresholded"),
        #         (sharpened, "Sharpened"),
        #         (inverted, "Inverted")
        #     ])

        return blurred_image

    def detect_problem_headers(self, image: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """
        Detect "Problem ##" headers in the image.

        Args:
            image: Preprocessed image

        Returns:
            List of tuples (x, y, w, h, problem_number)
        """
        # Use pytesseract to get detailed text information
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        print(data['text'])

        problem_headers = []

        # Look for "Problem" text followed by numbers
        for i, text in enumerate(data['text']):
            if text.strip():
                # Check if this looks like a problem header
                if re.search(r'problem', text.lower()) or text.lower().strip() == 'problem':
                    # Get bounding box
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    print(x, y, w, h)

                    # Try to extract problem number
                    problem_num = self._extract_problem_number(text, data, i)

                    if problem_num:
                        problem_headers.append((x, y, w, h, problem_num))

        # Sort by position (top to bottom, left to right)
        problem_headers.sort(key=lambda x: (x[1], x[0]))

        if self.debug:
            print(f"Found {len(problem_headers)} problem headers: {[p[4] for p in problem_headers]}")

        return problem_headers

    def _extract_problem_number(self, text: str, data: dict, index: int) -> Optional[int]:
        """
        Extract problem number from text or nearby text elements.

        Args:
            text: Current text element
            data: Full OCR data
            index: Index of current text element

        Returns:
            Problem number if found, None otherwise
        """
        # First try to extract from current text
        match = re.search(r'problem\s*(\d+)', text.lower())
        if match:
            return int(match.group(1))

        # If current text is just "Problem", look at nearby text elements
        if text.lower().strip() == 'problem':
            current_y = data['top'][index]
            current_x = data['left'][index]

            # Look for numbers in nearby positions
            for i, other_text in enumerate(data['text']):
                if i != index and other_text.strip():
                    other_y = data['top'][i]
                    other_x = data['left'][i]

                    # Check if it's roughly on the same line and nearby
                    if (abs(other_y - current_y) < 20 and
                            abs(other_x - current_x) < 200 and
                            other_text.strip().isdigit()):
                        return int(other_text.strip())

        return None

    def estimate_problem_regions(self, image: np.ndarray,
                                 headers: List[Tuple[int, int, int, int, int]]) -> List[Tuple[int, int, int, int, int]]:
        """
        Estimate the full region for each problem based on headers.

        Args:
            image: Preprocessed image
            headers: List of detected problem headers

        Returns:
            List of tuples (x, y, w, h, problem_number) for full problem regions
        """
        if not headers:
            return []

        height, width = image.shape
        regions = []

        # Estimate grid layout
        num_problems = len(headers)

        # Try to determine grid dimensions
        if num_problems <= 4:
            grid_rows, grid_cols = 2, 2
        elif num_problems <= 6:
            grid_rows, grid_cols = 2, 3
        elif num_problems <= 9:
            grid_rows, grid_cols = 3, 3
        else:
            # For larger numbers, estimate based on header positions
            unique_y = sorted(list(set([h[1] for h in headers])))
            grid_rows = len(unique_y)
            grid_cols = num_problems // grid_rows

        # Estimate cell dimensions
        cell_width = width // grid_cols
        cell_height = height // grid_rows

        for i, (hx, hy, hw, hh, prob_num) in enumerate(headers):
            # Determine grid position
            row = i // grid_cols
            col = i % grid_cols

            # Calculate region boundaries
            region_x = max(0, col * cell_width - self.padding)
            region_y = max(0, row * cell_height - self.padding)
            region_w = min(cell_width + 2 * self.padding, width - region_x)
            region_h = min(cell_height + 2 * self.padding, height - region_y)

            regions.append((region_x, region_y, region_w, region_h, prob_num))

        return regions

    def extract_and_save_problems(self, original_image: np.ndarray,
                                  regions: List[Tuple[int, int, int, int, int]]) -> None:
        """
        Extract problem regions and save them as separate images.

        Args:
            original_image: Original image (before preprocessing)
            regions: List of problem regions to extract
        """
        for x, y, w, h, prob_num in regions:
            # Extract region from original image
            problem_image = original_image[y:y + h, x:x + w]

            # Save the image
            filename = f"problem_images/problem-{prob_num:02d}.jpg"
            cv2.imwrite(filename, problem_image)

            if self.debug:
                print(f"Saved problem {prob_num} to {filename} "
                      f"(region: {x}, {y}, {w}, {h})")

    def _show_debug_images(self, images_and_titles: List[Tuple[np.ndarray, str]]) -> None:
        """Show debug images in a grid layout."""
        n_images = len(images_and_titles)
        cols = min(2, n_images)
        rows = (n_images + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))

        for i, (img, title) in enumerate(images_and_titles):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_detections(self, image: np.ndarray,
                             headers: List[Tuple[int, int, int, int, int]],
                             regions: List[Tuple[int, int, int, int, int]]) -> None:
        """
        Visualize detected headers and problem regions.

        Args:
            image: Original image
            headers: Detected problem headers
            regions: Estimated problem regions
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        # Draw headers in red
        for x, y, w, h, prob_num in headers:
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis_image, f"P{prob_num}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw regions in green
        for x, y, w, h, prob_num in regions:
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Problems (Red: Headers, Green: Regions)")
        plt.axis('off')
        plt.show()

    def process_page(self, image_path: str) -> None:
        """
        Process a complete page of Go problems.

        Args:
            image_path: Path to the input image
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        print(f"Processing image: {image_path}")
        print(f"Image dimensions: {original_image.shape}")

        # Preprocess image
        processed_image = self.preprocess_image(original_image)

        # Detect problem headers
        headers = self.detect_problem_headers(processed_image)

        if not headers:
            print("No problem headers detected!")
            return

        # Estimate problem regions
        regions = self.estimate_problem_regions(processed_image, headers)

        # Visualize detections if in debug mode
        if self.debug:
            self.visualize_detections(original_image, headers, regions)

        # Extract and save problems
        self.extract_and_save_problems(original_image, regions)

        print(f"Successfully extracted {len(regions)} problems")


def main():
    """Example usage of the Go problem extractor."""

    # Initialize extractor with debug mode enabled
    extractor = GoProblemExtractor(padding=30, debug=True)

    # Process an image
    image_path = "pages/page-63.jpg"  # Replace with your image path

    try:
        extractor.process_page(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")

        # If automatic detection fails, you can try manual parameter adjustment
        print("\nTry adjusting the following parameters:")
        print("- Increase padding if problems are cut off")
        print("- Check image quality and contrast")
        print("- Ensure problem headers contain 'Problem ##' text")


if __name__ == "__main__":
    main()
