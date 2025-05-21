"""
Main entry point for the Go SGF Converter application.
"""
import os
import argparse

from go_sgf_converter.core.converter import GoSGFConverter
from go_sgf_converter.io.image_loader import save_sgf
from go_sgf_converter.utils.debugger import Debugger


def main():
    """Main function to process Go problem images."""
    parser = argparse.ArgumentParser(description='Convert Go problem images to SGF format')
    parser.add_argument('--image', '-i', required=True, help='Path to the image file to process')
    parser.add_argument('--output', '-o', help='Output SGF file path')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Use current working directory (from where script is called)
    cwd = os.getcwd()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    image_basename = os.path.basename(args.image)
    problem_num = image_basename.split('-')[-1].split('.')[0]

    # Create SGFs directory in cwd if not specified
    if not args.output:
        sgf_dir = os.path.join(cwd, "SGFs")
        os.makedirs(sgf_dir, exist_ok=True)
        output_filename = os.path.join(sgf_dir, f"problem-{problem_num}.sgf")
    else:
        output_filename = os.path.join(cwd, args.output)

    # Create processed images directory: <cwd>/debug/<problem_num>
    processed_dir = os.path.join(cwd, "debug", str(problem_num))
    os.makedirs(processed_dir, exist_ok=True)

    # Create problem metadata directory in cwd
    metadata_dir = os.path.join(cwd, f"problem_metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Create Debugger singleton is debug argument is given
    Debugger.set_debug_enabled(args.debug)
    Debugger.get_instance(processed_dir)

    # Process image
    converter = GoSGFConverter()
    sgf_content = converter.process_image(args.image)
    
    # Save SGF file
    if sgf_content:
        save_sgf(sgf_content, output_filename)
        print(f"Processing complete: SGF file saved to {output_filename}")
    else:
        print("Processing failed: Could not generate SGF content")


if __name__ == "__main__":
    main()
