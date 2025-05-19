"""
Functions for detecting Go stones in board images.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union


def detect_stone_at_point(image: np.ndarray, point: Tuple[int, int]) -> Union[str, None]:
    """Detect stone at specific intersection.
    
    Args:
        image: Board image
        point: Coordinates of intersection (x, y)
        
    Returns:
        'black', 'white', or None if no stone detected
    """
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


def find_grid_parameters(h_lines: List, v_lines: List) -> Tuple:
    """Find grid corner and spacing using line detection.
    
    Args:
        h_lines: Horizontal lines
        v_lines: Vertical lines
        
    Returns:
        Dictionary containing 'corner', 'spacing_x', 'spacing_y', and 'corner_type'
    """
    # Find board corner and spacing
    if not h_lines or not v_lines:
        raise ValueError("No lines specified.")

    # Extract coordinates and calculate spacing as before
    v_coords = []
    for line in v_lines:
        x1, y1, x2, y2 = line
        v_coords.append((x1 + x2) // 2)

    h_coords = []
    for line in h_lines:
        x1, y1, x2, y2 = line
        h_coords.append((y1 + y2) // 2)

    # Sort coordinates
    v_coords.sort()
    h_coords.sort()

    # Calculate spacing
    def find_most_common_spacing(coords):
        if len(coords) < 2:
            return 30  # Default spacing
        differences = []
        # line_diffs = []
        for i in range(len(coords) - 1):
            diff = coords[i + 1] - coords[i]
            # line_diffs.append(diff)
            if 10 <= diff <= 65:  # Reasonable range for grid spacing
                differences.append(diff)
        # print(line_diffs)
        if differences:
            return int(np.median(differences))
        return 30

    spacing_x = find_most_common_spacing(v_coords)
    spacing_y = find_most_common_spacing(h_coords)

    print(f"DEBUG: Calculated spacing - X: {spacing_x}, Y: {spacing_y}")

    return spacing_x, spacing_y


def detect_board_corner(h_lines: List, v_lines: List, corners) -> str:
    """Detect which corner of the board is being shown.
    
    Args:
        h_lines: Horizontal lines
        v_lines: Vertical lines
        corners: Dictionary of detected corners
        
    Returns:
        Corner name string (e.g., 'top-right')
    """
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


def create_geometric_grid(board_image: np.ndarray, grid_params: Dict, board_coordinates: str, debugger) -> Dict[str, Tuple[int, int]]:
    """Create complete grid coordinates geometrically based on detected corner.
    
    Args:
        board_image: Image containing Go board
        grid_params: Dictionary with grid parameters
        board_coordinates: String of board coordinates (column labels)
        debugger: Debugger instance for saving intermediate images
        
    Returns:
        Dictionary mapping grid coordinates (e.g., 'q15') to image coordinates (x, y)
    """
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
                coord = f"{board_coordinates[board_col]}{board_row}"
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

    if debugger:
        debugger.save_debug_image(debug_image, "geometric_grid.jpg")

    print(f"DEBUG: Created {len(grid_points)} grid intersections for {corner_type} corner")

    return grid_points


def detect_stones_geometric(board_image: np.ndarray, h_lines, v_lines, corners, board_coordinates: str, debugger):
    """Detect stones using geometric grid calculation.
    
    Args:
        board_image: Image containing Go board
        h_lines: Detected horizontal lines
        v_lines: Detected vertical lines
        corners: Dictionary of board corners
        board_coordinates: String of board coordinates (column labels)
        debugger: Debugger instance for saving intermediate images
        
    Returns:
        Tuple of (stones, grid_points) where stones is {'black': [], 'white': []} and
        grid_points maps coordinates to points
    """
    print('detect_stones_geometric...')

    # Find grid parameters and corner
    corner_type = detect_board_corner(h_lines, v_lines, corners)
    grid_params = find_grid_parameters(board_image, h_lines, v_lines, corners, corner_type)

    # Create all grid coordinates geometrically
    grid_points = create_geometric_grid(board_image, grid_params, board_coordinates, debugger)

    # Detect stones at each intersection
    debug_image = board_image.copy()
    stones = {'black': [], 'white': []}

    for coord, point in grid_points.items():
        stone_color = detect_stone_at_point(board_image, point)
        if stone_color:
            stones[stone_color].append(coord)
            # Draw detected stone
            color = (0, 0, 0) if stone_color == 'black' else (255, 255, 255)
            cv2.circle(debug_image, point, 12, color, -1)
            cv2.circle(debug_image, point, 12, (255, 0, 0), 2)  # Red border
            # Add text label
            cv2.putText(debug_image, coord, (point[0] - 10, point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    if debugger:
        debugger.save_debug_image(debug_image, "detected_stones.jpg")

    print(f"DEBUG: Detected stones - Black: {stones['black']}, White: {stones['white']}")
    return stones#, grid_points
