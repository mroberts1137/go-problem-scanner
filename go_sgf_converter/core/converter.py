"""
Main converter class for transforming Go problem images to SGF format.
"""
from typing import Dict, Optional

import numpy as np
import json

from go_sgf_converter.utils.debugger import Debugger
from go_sgf_converter.utils import draw_utils as du
from go_sgf_converter.processing.image_segmenter import find_board_bounds_by_text
from go_sgf_converter.processing import text_extraction as te
from go_sgf_converter.processing import feature_extraction as fe
from go_sgf_converter.processing import board_transformations as bt
from go_sgf_converter.processing import stone_detection as sd
from go_sgf_converter.processing import image_processing as ip
from go_sgf_converter.io import image_loader, sgf_utils, serialize_data


class GoSGFConverter:
    """Main class for converting Go problem images to SGF format."""
    
    def __init__(self, processed_dir: str):
        """Initialize the converter.
        
        Args:
            processed_dir: Directory path to save debug images during processing
        """
        self.board_coordinates = 'abcdefghjklmnopqrst'  # Skipping 'i'
        self.processed_dir = processed_dir
        self.debugger = Debugger(processed_dir)
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
            self.debugger.save_debug_image(img, "original.jpg")
            
            # Process single problem
            problem_data = self.extract_problem_data(img)
            sgf = sgf_utils.create_sgf(problem_data)
            
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

        # Find text regions to isolate board
        problem_region, board_region, description_region, board_bounds = find_board_bounds_by_text(
            page_image, self.debugger)
        self.problem_metadata['board_bounds'] = board_bounds

        self.debugger.save_debug_image(problem_region, "problem_region.jpg")
        self.debugger.save_debug_image(description_region, "description_region.jpg")
        self.debugger.save_debug_image(board_region, "extracted_board.jpg")

        # Extract description, player, and problem number
        description, player = te.extract_description(description_region)
        problem_number = te.extract_problem_number(problem_region)
        self.problem_metadata['problem_number'] = problem_number

        # Extract board border box
        border_box = fe.get_board_border(board_region, self.debugger)
        self.problem_metadata['border_box'] = border_box

        # Orient board with projective transformation
        oriented_board = bt.orient_board(board_region, border_box)
        self.debugger.save_debug_image(oriented_board, "oriented_board.jpg")

        convex_hull_border = fe.get_board_border(oriented_board, self.debugger)

        inverted_oriented_board = ip.invert_image(oriented_board)
        thickened_oriented_board = ip.thicken_lines(inverted_oriented_board)
        self.debugger.save_debug_image(thickened_oriented_board, "thickened_oriented_board.jpg")

        #####################################################################################
        # Now oriented_board contains the centered-orthogonal board image
        #####################################################################################

        # Extract board lines
        oriented_edges, oriented_lines = fe.detect_lines(oriented_board)
        self.debugger.save_debug_image(oriented_edges, "oriented_board_edges.jpg")

        # Draw board lines
        oriented_line_image = du.draw_board_lines(oriented_board, oriented_lines)
        self.debugger.save_debug_image(oriented_line_image, "oriented_board_lines.jpg")

        # Classify horizontal/vertical lines
        h_lines, v_lines = fe.line_classifier(oriented_lines)

        # Get board line corners
        corners = fe.detect_board_corners(h_lines, v_lines)
        border_lines_image = du.draw_board_corners(oriented_board, corners)
        self.debugger.save_debug_image(border_lines_image, "border_lines_image.jpg")

        lines_border = fe.corners_to_array(corners)

        board_edges = fe.get_board_edges(convex_hull_border, lines_border)
        print(board_edges)

        # Determine grid spacing
        h_spacing, v_spacing = sd.find_grid_parameters(h_lines, v_lines)
        self.problem_metadata['grid_spacing'] = (h_spacing, v_spacing)

        filename = f'problem_metadata/problem_{problem_number}_metadata.json'

        serialize_data.save_to_json(self.problem_metadata, filename)

        #####################################################################################

        # invert image colors before contours

        exit(0)

        # Detect stones using geometric approach
        stones = sd.detect_stones_geometric(board_region, h_lines, v_lines,
                                                        border_box, self.board_coordinates, self.debugger)

        return {
            'stones': stones,
            'description': description,
            'player': player,
            'problem_number': problem_number
        }
