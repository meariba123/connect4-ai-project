import time
import random
import sys

ROWS, COLS = 6, 7  # Standard Connect Four board size

PLAYER_SYMBOLS = {
    0: " ",
    1: "\033[93m●\033[0m",  # Yellow
    2: "\033[91m●\033[0m"   # Red
}

def create_board():
    """Create a new empty game board"""
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    """Print the current game board state"""
    print("\n  " + "   ".join(str(i) for i in range(COLS)))
    print("  " + "—" * (COLS * 4 - 1))
    for row in board:
        print("| " + " | ".join(PLAYER_SYMBOLS[cell] for cell in row) + " |")
        print("  " + "—" * (COLS * 4 - 1))

def is_valid_column(col):
    """Check if a column number is valid"""
    try:
        return 0 <= int(col) < COLS
    except (ValueError, TypeError):
        return False

def is_valid_move(board, col):
    """Check if a move is valid in the current board state"""
    if not is_valid_column(col):
        return False
    col = int(col)
    return board[0][col] == 0  # Check if top row is empty

class Connect4:
    def __init__(self):
        """Initialize a new Connect4 game"""
        self.board = create_board()
        self.game_over = False
        self.winner = None
        self.current_player = 1  # Player 1 starts first

    def make_move(self, col, disc):
        """Attempt to make a move in the specified column"""
        try:
            col = int(col)
            if not (0 <= col < COLS):
                return False
                
            if not is_valid_move(self.board, col):
                return False
                
            # Find the first empty row in the column
            for row in reversed(range(ROWS)):
                if self.board[row][col] == 0:
                    self.board[row][col] = disc
                    
                    # Check for win or draw
                    if self.check_win(disc):
                        self.game_over = True
                        self.winner = disc
                    elif all(self.board[0][c] != 0 for c in range(COLS)):
                        self.game_over = True  # Draw
                    else:
                        self.current_player = 3 - disc  # Switch player
                    return True
            return False
        except (ValueError, TypeError):
            return False

    def check_win(self, disc):
        """Check if the specified player has won"""
        # Check horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                if all(self.board[row][col + i] == disc for i in range(4)):
                    return True
        # Check vertical
        for row in range(ROWS - 3):
            for col in range(COLS):
                if all(self.board[row + i][col] == disc for i in range(4)):
                    return True
        # Check diagonal (top-left to bottom-right)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if all(self.board[row + i][col + i] == disc for i in range(4)):
                    return True
        # Check diagonal (bottom-left to top-right)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                if all(self.board[row - i][col + i] == disc for i in range(4)):
                    return True
        return False

    def is_terminal(self):
        """Check if the game has ended"""
        return self.game_over

    def get_winner(self):
        """Get the winner of the game (1, 2, or None for draw)"""
        return self.winner

    def print_board(self):
        """Print the current board state"""
        print_board(self.board)

def play_human_vs_ai_game(ai_agent):
    game = Connect4()
    human_disc = 1
    ai_disc = 2

    print("\n=== Human vs AI ===")
    game.print_board()

    while not game.is_terminal():
        if game.current_player == human_disc:
            # Human move handling
            valid_move = False
            while not valid_move:
                try:
                    col = input("\nYour turn! Choose column (0-6): ").strip()
                    if col.lower() in ('quit', 'exit'):
                        print("Game exited by player.")
                        return
                    valid_move = game.make_move(col, human_disc)
                    if not valid_move:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter a number 0-6 or 'quit'")
        else:
            # Special handling for ML Agent
            if ai_agent.name == "MLAgent":
                print(f"\n{ai_agent.name} is thinking...", end='', flush=True)
                time.sleep(0.3)  # Brief pause for ML Agent
            else:
                print(f"\n{ai_agent.name} is thinking...")
            
            try:
                # Get AI move
                ai_move = ai_agent.select_move(game.board, ai_disc)
                
                # Handle ML Agent's dictionary return
                if isinstance(ai_move, dict):
                    ai_move = ai_move['move']
                
                # Clear line only for ML Agent
                if ai_agent.name == "MLAgent":
                    sys.stdout.write('\r' + ' ' * 50 + '\r')
                    sys.stdout.flush()
                
                if game.make_move(ai_move, ai_disc):
                    print(f"{ai_agent.name} placed in column {ai_move}")
                else:
                    print(f"AI made invalid move {ai_move}! Trying random...")
                    valid_cols = [c for c in range(COLS) if game.board[0][c] == 0]
                    ai_move = random.choice(valid_cols) if valid_cols else None
                    if ai_move is not None:
                        game.make_move(ai_move, ai_disc)
            except Exception as e:
                if ai_agent.name == "MLAgent":
                    sys.stdout.write('\r' + ' ' * 50 + '\r')
                    sys.stdout.flush()
                print(f"AI error: {str(e)}. Making random move...")
                valid_cols = [c for c in range(COLS) if game.board[0][c] == 0]
                ai_move = random.choice(valid_cols) if valid_cols else None
                if ai_move is not None:
                    game.make_move(ai_move, ai_disc)

        game.print_board()

    # Game over handling
    print("\nGame Over!")
    if game.get_winner() == human_disc:
        print("You win!")
    elif game.get_winner() == ai_disc:
        print(f" {ai_agent.name} wins!")
    else:
        print(" It's a draw! ")

    input("\nPress Enter to return to menu...")