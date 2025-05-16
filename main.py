import os
import GoSGFConverter

def main():
    problem_num = '23'
    model = '4b'

    # Process single JPG file
    image_path = f"problem_images/problem-{problem_num}.jpg"

    # Create SGFs directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sgf_dir = os.path.join(base_dir, "SGFs")
    os.makedirs(sgf_dir, exist_ok=True)

    # Create processed_images directory
    processed_dir = f"processed_images-{problem_num}-{model}"
    os.makedirs(processed_dir, exist_ok=True)

    converter = GoSGFConverter(processed_dir)

    sgf_file = converter.process_image(image_path)

    # Save SGF file
    if sgf_file:
        filename = os.path.join(sgf_dir, f"problem-{problem_num}-{model}.sgf")
        with open(filename, 'w') as f:
            f.write(sgf_file)
        print(f"Saved {filename}")

    print("Processing complete")


if __name__ == "__main__":
    main()
