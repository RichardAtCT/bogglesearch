# boggle_solver.py
import json
import logging
import sys

# Configure logging (optional)
logging.basicConfig(
    level=logging.ERROR,  # Set to ERROR to reduce verbosity on the web
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("boggle_solver.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_dictionary(filepath='words_dictionary.json'):
    try:
        with open(filepath, 'r') as f:
            words_list = json.load(f)
            words = set(word.upper() for word in words_list)
        return words
    except FileNotFoundError:
        logger.error(f"Dictionary file '{filepath}' not found.")
        return set()
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return set()

def build_prefixes(words):
    prefixes = set()
    for word in words:
        for i in range(1, len(word) + 1):
            prefixes.add(word[:i])
    return prefixes

def parse_board_input(board_input):
    elements = board_input.split(',')
    elements = [e.strip().upper() for e in elements]
    if len(elements) != 16:
        raise ValueError(f"Invalid board input. Expected 16 letters, got {len(elements)}.")
    for idx, letter in enumerate(elements):
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"Invalid character '{letter}' at position {idx+1}. Each entry must be a single alphabetic character.")
    board = [elements[i*4:(i+1)*4] for i in range(4)]
    return board

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
]

def in_bounds(r, c):
    return 0 <= r < 4 and 0 <= c < 4

def dfs(r, c, prefix, visited, valid_prefixes, valid_words, found, board):
    prefix += board[r][c]
    if prefix not in valid_prefixes:
        return
    if prefix in valid_words and len(prefix) >= 4:
        found.add(prefix)
    visited[r][c] = True
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and not visited[nr][nc]:
            dfs(nr, nc, prefix, visited, valid_prefixes, valid_words, found, board)
    visited[r][c] = False

def find_words(board, valid_words, valid_prefixes):
    found_words = set()
    visited = [[False] * 4 for _ in range(4)]
    for row in range(4):
        for col in range(4):
            dfs(row, col, prefix="", visited=visited,
                valid_prefixes=valid_prefixes,
                valid_words=valid_words,
                found=found_words,
                board=board)
    sorted_words = sorted(found_words, key=lambda x: (len(x), x))
    return sorted_words