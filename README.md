### Go Problem Scanner

This package takes a scanned image of a go problem and converts it to an sgf file.

In its current version it requires an image to have the following structure:

![Problem image](/problem_images/problem-23.jpg)

To run from the command line:

```bash
python -m go_sgf_converter.main --image <image.jpg>
```

CLI arguments:

```bash
--image, -i: 'Path to the image file to process'
--output, -o: 'Output SGF file path', default="./SGFs/problem-{problem_num}.sgf"
--debug, -d: 'Enable debug output'
```