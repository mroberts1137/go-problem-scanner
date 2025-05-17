"""
Utilities for creating SGF files from Go problem data.
"""
from typing import Dict


def create_sgf(problem_data: Dict) -> str:
    """Create SGF string from problem data.
    
    Args:
        problem_data: Dictionary containing problem information
        
    Returns:
        SGF formatted string representation of the problem
    """
    print('create_sgf...')
    print(f"DEBUG: Creating SGF with data: {problem_data}")

    # Start with required properties
    sgf_parts = ["(;GM[1]FF[4]SZ[19]"]

    # Add problem number
    if problem_data.get('problem_number'):
        sgf_parts.append(f"GN[Problem {problem_data['problem_number']}]")

    # Add player to move
    if problem_data.get('player'):
        sgf_parts.append(f"PL[{problem_data['player']}]")

    # Add black stones
    if problem_data.get('stones', {}).get('black'):
        black_coords = ''.join(f"[{coord}]" for coord in problem_data['stones']['black'])
        sgf_parts.append(f"AB{black_coords}")

    # Add white stones
    if problem_data.get('stones', {}).get('white'):
        white_coords = ''.join(f"[{coord}]" for coord in problem_data['stones']['white'])
        sgf_parts.append(f"AW{white_coords}")

    # Add comment with description
    if problem_data.get('description'):
        comment = problem_data['description'].replace(']', '\\]')
        sgf_parts.append(f"C[{comment}]")

    sgf_parts.append(")")

    sgf_content = ''.join(sgf_parts)
    print(f"DEBUG: Generated SGF: {sgf_content}")
    return sgf_content
