### Go Problem Scanner

This package takes a scanned image of a go problem and converts it to an sgf file.

In its current version it requires an image to have the following structure:

![Problem image](/problem_images/problem-23.jpg)

It will output an sgf with basic fields:

```txt
(;GM[1]FF[4]SZ[19]GN[Problem 23]
PL[B]AB[nb][ic][lc][oc][pc]AW[mb][nc][qc][nd][qd][qg]
C[How can Black escape with his stones on the right?])
```

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