sgf_coordinates = 'abcdefghijklmnopqrst'


def convert_coord_to_sgf_coord(coord):
    return f'{sgf_coordinates[coord[1]]}{sgf_coordinates[coord[0]]}'


def convert_sgf_coord_to_coord(sgf_coord: str):
    """
    Converts an SGF coordinate string like 'dd' back to a (row, col) tuple.

    Parameters:
        sgf_coord (str): Two-character SGF coordinate string.

    Returns:
        tuple: (row, col) as integers.
    """
    row = sgf_coordinates.index(sgf_coord[1])
    col = sgf_coordinates.index(sgf_coord[0])
    return row, col
