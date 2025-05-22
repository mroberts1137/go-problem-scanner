"""
Main converter class for transforming Go problem images to SGF format.
"""
from typing import Dict, Optional

import numpy as np

from go_sgf_converter.utils.debugger import Debugger
from go_sgf_converter.utils import draw_utils as du
from go_sgf_converter.processing import image_segmenter
from go_sgf_converter.processing import text_extraction as te
from go_sgf_converter.processing import feature_extraction as fe
from go_sgf_converter.processing import board_transformations as bt
from go_sgf_converter.processing import stone_detection as sd
from go_sgf_converter.processing import image_processing as ip
from go_sgf_converter.io import image_loader, save_sgf, serialize_data


class GoSGFConverter:
    """Main class for converting Go problem images to SGF format."""

    def __init__(self):
        """Initialize the converter.
        """
        self.debugger = Debugger.get_instance()
        self.problem_metadata = {}

    def process_image(self, image_path: str) -> Optional[str]:
        """Main function to process a single JPG image and return SGF string.

        Args:
            image_path: Path to image file

        Returns:
            SGF formatted string representation of the problem, or None if failed
        """
        print('process_image...')

        try:
            img = image_loader.load_image(image_path)

            # Save original image
            if self.debugger:
                self.debugger.save_debug_image(img, "original.jpg")

            # Process single problem
            problem_data = self.extract_problem_data(img)
            sgf = save_sgf.create_sgf(problem_data)

            return sgf
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    def extract_problem_data(self, page_image: np.ndarray) -> Dict:
        """Extract components from a single problem image.

        Args:
            page_image: Image containing Go problem

        Returns:
            Dictionary with problem data including stones, description and player
        """
        print('extract_problem_data...')

        # TODO: if problem_metadata exists, bypass processing steps
        # TODO: features and oriented_features don't need the same features extraction function
        # TODO: line spacing counting nearest neighbor should try to count spacing between nearest medians

        # Find text regions to isolate board
        problem_region, board_region, description_region, board_bounds = image_segmenter.segment_board_components(
            page_image)
        self.problem_metadata['board_bounds'] = board_bounds

        if self.debugger:
            self.debugger.save_debug_image(problem_region, "problem_region.jpg")
            self.debugger.save_debug_image(description_region, "description_region.jpg")
            self.debugger.save_debug_image(board_region, "extracted_board.jpg")

        # Extract description, player, and problem number
        description, player = te.extract_description(description_region)
        problem_number = te.extract_problem_number(problem_region)
        self.problem_metadata['problem_number'] = problem_number

        # Extract board features
        features = fe.extract_board_features(board_region)

        self.problem_metadata['proj_mat_corners'] = features['border_box']

        print('Orienting board with ', features['border_box'])

        # Orient board with projective transformation
        oriented_board = bt.orient_board(board_region, features['border_box'])

        # Get features for oriented board
        oriented_features = fe.extract_board_features(oriented_board)

        inverted_oriented_board = ip.invert_image(oriented_board)
        thickened_oriented_board = ip.thicken_lines(inverted_oriented_board)

        if self.debugger:
            self.debugger.save_debug_image(thickened_oriented_board, "thickened_oriented_board.jpg")

            oriented_line_image = du.draw_board_lines(oriented_board, oriented_features['lines'])
            self.debugger.save_debug_image(oriented_line_image, "oriented_board_lines.jpg")

            # Draw border lines
            border_lines_image = du.draw_board_corners(oriented_board, oriented_features['corners'])
            self.debugger.save_debug_image(border_lines_image, "border_lines_image.jpg")

            # Draw inner/outer bounding regions used in determining board edges
            extended_region_image = du.draw_inner_outer_borders(
                oriented_board,
                oriented_features['lines_border'],
                oriented_features['border_box']
            )
            self.debugger.save_debug_image(extended_region_image, "extended_region_image.jpg")

        self.problem_metadata['corner_points'] = oriented_features['corner_points']

        # Build the board grid
        full_board_grid = fe.build_board_grid(oriented_features)

        # Image grid filters out grid points outside the image
        image_grid = full_board_grid['image_grid']
        self.problem_metadata['grid_spacing'] = full_board_grid['grid_spacing']

        # Detect stones at image_grid intersection points
        stones = sd.detect_stones_geometric(thickened_oriented_board, image_grid, full_board_grid['grid_spacing'])

        # Visualize grid
        if self.debugger:
            grid_image = du.draw_grid_points(oriented_board, image_grid)
            self.debugger.save_debug_image(grid_image, "grid_visualization.jpg")

            stones_image = du.draw_stones_on_board(grid_image, stones, image_grid)
            self.debugger.save_debug_image(stones_image, "stones_image.jpg")

        print(self.problem_metadata)

        # Save problem metadata
        filename = f'problem_metadata/problem_{problem_number}_metadata.json'
        serialize_data.save_to_json(self.problem_metadata, filename)

        return {
            'stones': stones,
            'description': description,
            'player': player,
            'problem_number': problem_number
        }
