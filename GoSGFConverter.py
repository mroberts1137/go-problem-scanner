import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict

from debugger import Debugger
import board_transformations as bt
from image_segmenter import find_board_bounds_by_text
import text_extraction as te
import feature_extraction as fe


class GoSGFConverter:
    def __init__(self, processed_dir):
        self.board_coordinates = 'abcdefghjklmnopqrst'  # Skipping 'i'
        self.processed_dir = processed_dir
        self.debug_count = 0
        self.debugger = Debugger(processed_dir)

    def process_image(self, image_path: str) -> str:
        """Main function to process a single JPG image and return SGF string"""
        print('process_image...')
        img = self.load_jpg_image(image_path)

        # Save original image
        self.debugger.save_debug_image(img, "original.jpg")

        # Process single problem
        problem_data = self.extract_problem_data(img)
        sgf = self.create_sgf(problem_data)

        return sgf

    def load_jpg_image(self, image_path: str) -> np.ndarray:
        """Load a single JPG image"""
        print('load_jpg_image...')
        img = Image.open(image_path).convert('RGB')
        return np.array(img)

    def extract_problem_data(self, page_image: np.ndarray) -> Dict:
        """Extract components from a single problem image"""
        print('extract_problem_data...')

        # Find text regions to isolate board
        problem_region, board_region, description_region = find_board_bounds_by_text(page_image, self.debugger)

        # Extract description, player, and problem number
        description, player = te.extract_description(description_region)
        problem_number = te.extract_problem_number(problem_region)

        # Extract board lines
        edges, lines = fe.detect_lines(board_region)
        self.debugger.save_debug_image(edges, "board_edges.jpg")

        # Draw board lines
        line_image = fe.draw_board_lines(board_region, lines)
        self.debugger.save_debug_image(line_image, "board_lines.jpg")

        # Classify horizontal/vertical lines
        h_lines, v_lines = fe.line_classifier(lines)

        # Detect board corners
        corners = fe.detect_board_corners(h_lines, v_lines)

        # Draw board corners
        corners_image = fe.draw_board_corners(board_region, corners)
        self.debugger.save_debug_image(corners_image, "board_corners.jpg")

        # Orient board with projective transformation
        oriented_board = bt.orient_board(board_region, corners)
        self.debugger.save_debug_image(oriented_board, "oriented_board.jpg")

        #####################################################################################

        # Now oriented_board contains the centered-orthogonal board image

        # Extract board lines
        oriented_edges, oriented_lines = fe.detect_lines(oriented_board)
        self.debugger.save_debug_image(oriented_edges, "oriented_board_edges.jpg")

        # Draw board lines
        oriented_line_image = fe.draw_board_lines(oriented_board, oriented_lines)
        self.debugger.save_debug_image(oriented_line_image, "oriented_board_lines.jpg")

        #####################################################################################

        # Everything past this point is WIP

        # Detect stones using geometric approach
        stones, grid_overlay = self.detect_stones_geometric(board_region, h_lines, v_lines, corners)

        return {
            'stones': stones,
            'description': description,
            'player': player,
            'problem_number': problem_number
        }


    def detect_stones_geometric(self, board_image: np.ndarray, h_lines, v_lines, corners) -> Tuple[Dict[str, List[str]], Dict]:
        """Detect stones using geometric grid calculation"""
        print('detect_stones_geometric...')

        # Find grid parameters and corner
        grid_params = self.find_grid_parameters(board_image, h_lines, v_lines, corners)

        # Create all grid coordinates geometrically
        grid_points = self.create_geometric_grid(board_image, grid_params)

        # Detect stones at each intersection
        debug_image = board_image.copy()
        stones = {'black': [], 'white': []}

        for coord, point in grid_points.items():
            stone_color = self.detect_stone_at_point(board_image, point)
            if stone_color:
                stones[stone_color].append(coord)
                # Draw detected stone
                color = (0, 0, 0) if stone_color == 'black' else (255, 255, 255)
                cv2.circle(debug_image, point, 12, color, -1)
                cv2.circle(debug_image, point, 12, (255, 0, 0), 2)  # Red border
                # Add text label
                cv2.putText(debug_image, coord, (point[0] - 10, point[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        self.debugger.save_debug_image(debug_image, "detected_stones.jpg")

        print(f"DEBUG: Detected stones - Black: {stones['black']}, White: {stones['white']}")
        return stones, grid_points

    def find_grid_parameters(self, board_image: np.ndarray, h_lines, v_lines, corners) -> Dict:
        """Find grid corner and spacing using line detection"""
        # Detect which corner is being shown
        corner_type = self.detect_board_corner(h_lines, v_lines, corners)

        # Find board corner and spacing
        corner, spacing = self.calculate_grid_from_lines(h_lines, v_lines, board_image.shape, corner_type)

        return {
            'corner': corner,
            'spacing_x': spacing[0],
            'spacing_y': spacing[1],
            'corner_type': corner_type
        }

    def detect_board_corner(self, h_lines: List, v_lines: List, corners) -> str:
        """Detect which corner of the board is being shown"""
        print('detect_board_corner...')

        if not h_lines or not v_lines:
            print("DEBUG: No lines detected, defaulting to top-right corner")
            return 'top-right'


        # Check which corner has no lines extending beyond it
        for corner_name, (cx, cy) in corners.items():
            has_extensions = False

            # Check if lines extend beyond this corner
            if corner_name.startswith('top'):
                # Check if any horizontal line extends above this point
                for line in h_lines:
                    line_y = min(line[1], line[3])
                    if line_y < cy - 10:  # 10px tolerance
                        has_extensions = True
                        break
            else:  # bottom
                # Check if any horizontal line extends below this point
                for line in h_lines:
                    line_y = max(line[1], line[3])
                    if line_y > cy + 10:  # 10px tolerance
                        has_extensions = True
                        break

            if not has_extensions:
                if corner_name.endswith('left'):
                    # Check if any vertical line extends left of this point
                    for line in v_lines:
                        line_x = min(line[0], line[2])
                        if line_x < cx - 10:  # 10px tolerance
                            has_extensions = True
                            break
                else:  # right
                    # Check if any vertical line extends right of this point
                    for line in v_lines:
                        line_x = max(line[0], line[2])
                        if line_x > cx + 10:  # 10px tolerance
                            has_extensions = True
                            break

            if not has_extensions:
                print(f"DEBUG: Detected board corner: {corner_name}")
                return corner_name

        # Default to top-right if no clear corner found
        print("DEBUG: No clear corner detected, defaulting to top-right")
        return 'top-right'


    def calculate_grid_from_lines(self, h_lines: List, v_lines: List, image_shape: Tuple, corner_type: str):
        """Calculate grid corner and spacing from detected lines"""
        print('calculate_grid_from_lines...')

        if not h_lines or not v_lines:
            # Fallback to default values based on corner type
            print("DEBUG: No lines detected, using default grid parameters")
            height, width = image_shape[:2]
            if corner_type == 'top-left':
                return (50, 50), (30, 30)
            elif corner_type == 'top-right':
                return (width - 300, 50), (30, 30)
            elif corner_type == 'bottom-left':
                return (50, height - 300), (30, 30)
            else:  # bottom-right
                return (width - 300, height - 300), (30, 30)

        # Extract coordinates and calculate spacing as before
        v_x_coords = []
        for line in v_lines:
            x1, y1, x2, y2 = line
            v_x_coords.append((x1 + x2) // 2)

        h_y_coords = []
        for line in h_lines:
            x1, y1, x2, y2 = line
            h_y_coords.append((y1 + y2) // 2)

        # Sort coordinates
        v_x_coords.sort()
        h_y_coords.sort()

        # Calculate spacing
        def find_most_common_spacing(coords):
            if len(coords) < 2:
                return 30  # Default spacing
            differences = []
            for i in range(len(coords) - 1):
                diff = coords[i + 1] - coords[i]
                if 15 <= diff <= 50:  # Reasonable range for grid spacing
                    differences.append(diff)
            if differences:
                return int(np.median(differences))
            return 30

        spacing_x = find_most_common_spacing(v_x_coords)
        spacing_y = find_most_common_spacing(h_y_coords)

        print(f"DEBUG: Calculated spacing - X: {spacing_x}, Y: {spacing_y}")

        # Find corner based on detected corner type
        if corner_type == 'top-left':
            corner_x = min(v_x_coords) if v_x_coords else 50
            corner_y = min(h_y_coords) if h_y_coords else 50
        elif corner_type == 'top-right':
            corner_x = max(v_x_coords) if v_x_coords else image_shape[1] - 50
            corner_y = min(h_y_coords) if h_y_coords else 50
        elif corner_type == 'bottom-left':
            corner_x = min(v_x_coords) if v_x_coords else 50
            corner_y = max(h_y_coords) if h_y_coords else image_shape[0] - 50
        else:  # bottom-right
            corner_x = max(v_x_coords) if v_x_coords else image_shape[1] - 50
            corner_y = max(h_y_coords) if h_y_coords else image_shape[0] - 50

        print(f"DEBUG: {corner_type} corner at ({corner_x}, {corner_y})")

        return (corner_x, corner_y), (spacing_x, spacing_y)

    def create_geometric_grid(self, board_image: np.ndarray, grid_params: Dict) -> Dict[str, Tuple[int, int]]:
        """Create complete grid coordinates geometrically based on detected corner"""
        print('create_geometric_grid...')

        corner_x, corner_y = grid_params['corner']
        spacing_x = grid_params['spacing_x']
        spacing_y = grid_params['spacing_y']
        corner_type = grid_params['corner_type']

        height, width = board_image.shape[:2]
        grid_points = {}
        debug_image = board_image.copy()

        # Calculate grid direction based on corner type
        if corner_type == 'top-left':
            x_direction = 1  # Move right
            y_direction = 1  # Move down
            x_range = range(0, (width - corner_x) // spacing_x)
            y_range = range(0, (height - corner_y) // spacing_y)
        elif corner_type == 'top-right':
            x_direction = -1  # Move left
            y_direction = 1  # Move down
            x_range = range(0, (corner_x) // spacing_x + 1)
            y_range = range(0, (height - corner_y) // spacing_y)
        elif corner_type == 'bottom-left':
            x_direction = 1  # Move right
            y_direction = -1  # Move up
            x_range = range(0, (width - corner_x) // spacing_x)
            y_range = range(0, (corner_y) // spacing_y + 1)
        else:  # bottom-right
            x_direction = -1  # Move left
            y_direction = -1  # Move up
            x_range = range(0, (corner_x) // spacing_x + 1)
            y_range = range(0, (corner_y) // spacing_y + 1)

        # Map to board coordinates based on corner type
        for i, x_idx in enumerate(x_range):
            for j, y_idx in enumerate(y_range):
                x = corner_x + x_idx * spacing_x * x_direction
                y = corner_y + y_idx * spacing_y * y_direction

                # Convert to board coordinates
                if corner_type == 'top-left':
                    board_col = i
                    board_row = 19 - j  # Row 19 is at top
                elif corner_type == 'top-right':
                    board_col = 18 - i  # Column 't' (18) is on right
                    board_row = 19 - j  # Row 19 is at top
                elif corner_type == 'bottom-left':
                    board_col = i
                    board_row = j + 1  # Row 1 is at bottom
                else:  # bottom-right
                    board_col = 18 - i  # Column 't' (18) is on right
                    board_row = j + 1  # Row 1 is at bottom

                # Ensure coordinates are valid
                if 0 <= board_col < 19 and 1 <= board_row <= 19:
                    coord = f"{self.board_coordinates[board_col]}{board_row}"
                    grid_points[coord] = (x, y)

                    # Draw intersection point
                    cv2.circle(debug_image, (x, y), 2, (255, 0, 0), -1)

                    # Draw coordinate label for some intersections
                    if i % 2 == 0 and j % 2 == 0:
                        cv2.putText(debug_image, coord, (x - 15, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Draw grid lines
        for x_idx in x_range:
            x = corner_x + x_idx * spacing_x * x_direction
            y1 = corner_y
            y2 = corner_y + (len(y_range) - 1) * spacing_y * y_direction
            cv2.line(debug_image, (x, y1), (x, y2), (0, 255, 0), 1)

        for y_idx in y_range:
            y = corner_y + y_idx * spacing_y * y_direction
            x1 = corner_x
            x2 = corner_x + (len(x_range) - 1) * spacing_x * x_direction
            cv2.line(debug_image, (x1, y), (x2, y), (0, 255, 0), 1)

        # Draw corner indicator
        cv2.circle(debug_image, (corner_x, corner_y), 10, (0, 0, 255), 3)
        cv2.putText(debug_image, corner_type, (corner_x - 50, corner_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        self.debugger.save_debug_image(debug_image, "geometric_grid.jpg")
        print(f"DEBUG: Created {len(grid_points)} grid intersections for {corner_type} corner")

        return grid_points

    def detect_stone_at_point(self, image: np.ndarray, point: Tuple[int, int]) -> str:
        """Detect stone at specific intersection"""
        x, y = point
        radius = 15  # Detection radius

        # Ensure we don't go out of bounds
        height, width = image.shape[:2]
        if x - radius < 0 or x + radius >= width or y - radius < 0 or y + radius >= height:
            return None

        # Extract region around intersection
        region = image[y - radius:y + radius, x - radius:x + radius]
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        # Calculate average intensity
        avg_intensity = np.mean(gray_region)

        # Look for circular patterns
        circles = cv2.HoughCircles(gray_region, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=15, minRadius=8, maxRadius=18)

        if circles is not None and len(circles[0]) > 0:
            # Found circular pattern, determine color based on intensity
            if avg_intensity < 100:  # Dark threshold for black stones
                return 'black'
            elif avg_intensity > 150:  # Light threshold for white stones
                return 'white'

        return None


    def create_sgf(self, problem_data: Dict) -> str:
        """Create SGF string from problem data"""
        print('create_sgf...')
        print(f"DEBUG: Creating SGF with data: {problem_data}")

        # Start with required properties
        sgf_parts = ["(;GM[1]FF[4]SZ[19]"]

        # Add problem number
        sgf_parts.append(f"GN[Problem {problem_data['problem_number']}]")

        # Add player to move
        sgf_parts.append(f"PL[{problem_data['player']}]")

        # Add black stones
        if problem_data['stones']['black']:
            black_coords = ''.join(f"[{coord}]" for coord in problem_data['stones']['black'])
            sgf_parts.append(f"AB{black_coords}")

        # Add white stones
        if problem_data['stones']['white']:
            white_coords = ''.join(f"[{coord}]" for coord in problem_data['stones']['white'])
            sgf_parts.append(f"AW{white_coords}")

        # Add comment with description
        if problem_data['description']:
            comment = problem_data['description'].replace(']', '\\]')
            sgf_parts.append(f"C[{comment}]")

        sgf_parts.append(")")

        sgf_content = ''.join(sgf_parts)
        print(f"DEBUG: Generated SGF: {sgf_content}")
        return sgf_content
