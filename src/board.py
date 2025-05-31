import random
import numpy as np
from typing import List, Tuple, Optional

class Board:
    def __init__(self, size: int = 4):
        self.size = size
        self.score = 0
        self.board = np.zeros((size, size), dtype=int)
        # Initialize board with two random tiles
        self._add_new_tile()
        self._add_new_tile()
    
    def _add_new_tile(self) -> bool:
        """Add a new tile (2 or 4) to a random empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        
        row, col = random.choice(empty_cells)
        # 90% chance for 2, 10% chance for 4
        self.board[row, col] = 2 if random.random() < 0.9 else 4
        return True
    
    def _compress(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compress the line by moving all non-zero numbers to the left."""
        non_zero = line[line != 0]
        merged_line = np.zeros_like(line)
        merged_line[:len(non_zero)] = non_zero
        return merged_line, 0
    
    def _merge(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """Merge adjacent equal numbers and calculate score."""
        score = 0
        for i in range(len(line) - 1):
            if line[i] != 0 and line[i] == line[i + 1]:
                line[i] *= 2
                line[i + 1] = 0
                score += line[i]
        return line, score
    
    def move(self, direction: str) -> bool:
        """
        Move tiles in the specified direction ('up', 'down', 'left', 'right').
        Returns True if the move resulted in any change.
        """
        original_board = self.board.copy()
        
        # Convert direction to number of rotations
        rotations = {
            'left': 0,
            'up': 1,
            'right': 2,
            'down': 3
        }
        
        if direction not in rotations:
            raise ValueError("Invalid direction. Must be 'up', 'down', 'left', or 'right'")
        
        # Rotate board to handle all directions as 'left'
        self.board = np.rot90(self.board, k=rotations[direction])
        
        # Process each row
        moved = False
        for i in range(self.size):
            # Compress
            self.board[i], _ = self._compress(self.board[i])
            # Merge
            self.board[i], score = self._merge(self.board[i])
            # Compress again after merging
            self.board[i], _ = self._compress(self.board[i])
            self.score += score
            
        # Rotate back
        self.board = np.rot90(self.board, k=-rotations[direction])
        
        # Check if the board changed
        if not np.array_equal(original_board, self.board):
            self._add_new_tile()
            moved = True
            
        return moved
    
    def is_game_over(self) -> bool:
        """Check if no moves are possible."""
        if np.any(self.board == 0):
            return False
            
        # Check for possible merges
        for i in range(self.size):
            for j in range(self.size - 1):
                # Check horizontal merges
                if self.board[i][j] == self.board[i][j + 1]:
                    return False
                # Check vertical merges
                if self.board[j][i] == self.board[j + 1][i]:
                    return False
        return True
    
    def has_won(self) -> bool:
        """Check if 2048 tile is present."""
        return np.any(self.board >= 2048)
    
    def get_board(self) -> np.ndarray:
        """Return the current board state."""
        return self.board.copy()
    
    def get_score(self) -> int:
        """Return the current score."""
        return self.score