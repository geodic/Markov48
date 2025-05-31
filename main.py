from src.board import Board

def print_board(board):
    """Pretty print the game board."""
    for row in board:
        print(' '.join(f'{cell:4d}' for cell in row))
    print()

def main():
    # Initialize the game
    game = Board()
    
    print("Welcome to 2048!")
    print("Use 'w' (up), 's' (down), 'a' (left), 'd' (right) to move tiles")
    print("Press 'q' to quit\n")
    
    # Game loop
    while True:
        print_board(game.get_board())
        print(f"Score: {game.get_score()}\n")
        
        # Get user input
        move = input("Enter your move: ").lower()
        
        if move == 'q':
            print("Thanks for playing!")
            break
            
        # Convert WASD to directions
        direction_map = {
            'w': 'up',
            's': 'down',
            'a': 'left',
            'd': 'right'
        }
        
        if move in direction_map:
            moved = game.move(direction_map[move])
            if not moved:
                print("Invalid move!")
            
            if game.has_won():
                print("\nCongratulations! You've reached 2048!")
                choice = input("Continue playing? (y/n): ")
                if choice.lower() != 'y':
                    break
                    
            if game.is_game_over():
                print("\nGame Over! No more moves possible.")
                break
        else:
            print("Invalid input! Use 'w', 'a', 's', 'd' to move, 'q' to quit")

if __name__ == "__main__":
    main()
