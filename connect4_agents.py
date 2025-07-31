# Import necessary libraries for the Connect Four AI agents
# References:
# - Python Standard Library: https://docs.python.org/3/library/
# - NumPy: https://numpy.org/doc/
# - Pandas: https://pandas.pydata.org/docs/
# - Joblib: https://joblib.readthedocs.io/en/latest/
import math
import random
import time
import copy
import tracemalloc
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Import custom performance utilities
from performance_utils import PerformanceUtils

# Game board dimensions (standard Connect Four is 6 rows x 7 columns)
COLS = PerformanceUtils.COLS  # Number of columns
ROWS = PerformanceUtils.ROWS  # Number of rows
AI_DEPTH = 4  # Default depth for Minimax algorithm

class RandomAgent:
    """
    A simple agent that makes random valid moves.
    Reference: Russell & Norvig, "Artificial Intelligence: A Modern Approach" - Simple reflex agents
    """
    def __init__(self, symbol=None):
        """
        Initialize the agent with a player symbol (1 or 2)
        """
        self.symbol = symbol  # Player's disc symbol (1 or 2)
        self.name = 'Random'  # Agent name for identification

    def select_move(self, board, disc):
        """
        Select a random valid move from available columns
        Args:
            board: Current game board state
            disc: Player's disc (1 or 2)
        Returns:
            int: Column index for the move
        """
        valid_cols = [c for c in range(COLS) if board[0][c] == 0]  # Find columns with space
        return random.choice(valid_cols)  # Return random valid column

    def get_move(self, board):
        """
        Public method to get the agent's move
        Args:
            board: Current game board state
        Returns:
            int: Selected column index
        """
        return self.select_move(board, self.symbol)


class SmartAgent:
    """
    A rule-based agent with basic Connect Four strategy.
    Reference: 
    - Allis, L.V. (1994). "A Knowledge-based Approach of Connect-Four"
    - Winning Moves Games (1988). "Connect Four game instructions"
    """
    def __init__(self, symbol=None):
        """
        Initialize the agent with a player symbol (1 or 2)
        """
        self.symbol = symbol  # Player's disc symbol (1 or 2)
        self.name = 'Smart'   # Agent name for identification

    def select_move(self, board, disc):
        """
        Select a move using basic Connect Four strategy:
        1. Check for immediate win
        2. Block opponent's immediate win
        3. Prefer center column
        4. Fall back to random move
        Args:
            board: Current game board state
            disc: Player's disc (1 or 2)
        Returns:
            int: Column index for the move
        """
        valid_cols = [c for c in range(COLS) if board[0][c] == 0]  # Valid columns
        opponent_disc = 1 if disc == 2 else 2  # Determine opponent's disc

        # Check for immediate win (highest priority)
        for col in valid_cols:
            row = self.get_next_open_row(board, col)
            temp_board = [r[:] for r in board]  # Create board copy
            temp_board[row][col] = disc  # Simulate move
            if self.check_win(temp_board, disc):
                return col  # Take winning move

        # Block opponent's immediate win (second priority)
        for col in valid_cols:
            row = self.get_next_open_row(board, col)
            temp_board = [r[:] for r in board]  # Create board copy
            temp_board[row][col] = opponent_disc  # Simulate opponent move
            if self.check_win(temp_board, opponent_disc):
                return col  # Block opponent's win

        # Prefer center column (strategic advantage)
        if 3 in valid_cols:  # Center column (index 3 in 0-based)
            return 3
            
        return random.choice(valid_cols)  # Fallback to random move

    def get_move(self, board):
        """
        Public method to get the agent's move
        Args:
            board: Current game board state
        Returns:
            int: Selected column index
        """
        return self.select_move(board, self.symbol)

    def get_next_open_row(self, board, col):
        """
        Find the next empty row in a given column
        Args:
            board: Current game board state
            col: Column index to check
        Returns:
            int: Row index of first empty space
        """
        for r in range(ROWS - 1, -1, -1):  # Check from bottom up
            if board[r][col] == 0:
                return r

    def check_win(self, board, disc):
        """
        Check if the given disc has a winning position
        Args:
            board: Current game board state
            disc: Player's disc (1 or 2)
        Returns:
            bool: True if disc has won, False otherwise
        """
        # Horizontal check
        for c in range(COLS - 3):
            for r in range(ROWS):
                if all(board[r][c + i] == disc for i in range(4)):
                    return True
        # Vertical check
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(board[r + i][c] == disc for i in range(4)):
                    return True
        # Diagonal / check
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                if all(board[r + i][c + i] == disc for i in range(4)):
                    return True
        # Diagonal \ check
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if all(board[r - i][c + i] == disc for i in range(4)):
                    return True
        return False


class MinimaxAgent:
    """
    Minimax algorithm with alpha-beta pruning for Connect Four.
    References:
    - Russell & Norvig, "Artificial Intelligence: A Modern Approach" - Adversarial Search
    - Knuth, D. E. and Moore, R. W. (1975). "An analysis of alpha-beta pruning"
    - Allis, L.V. (1988). "A Knowledge-based Approach of Connect-Four"
    """
    def __init__(self, symbol=None, max_depth=4, time_limit=5.0):
        """
        Initialize the Minimax agent with:
        - Player symbol
        - Maximum search depth
        - Time limit for move calculation
        """
        self.symbol = symbol  # Player's disc (1 or 2)
        self.max_depth = max_depth  # Maximum search depth
        self.time_limit = time_limit  # Time limit per move (seconds)
        self.pruning_count = 0  # Count of alpha-beta prunings
        self.name = 'MiniMax'  # Agent name
        # Performance metrics tracking
        self.metrics = {
            'nodes_expanded': 0,
            'max_depth_reached': 0,
            'execution_times': [],
            'memory_usages': [],
            'pruning_counts': 0,
            'iterative_deepening_depths': []
        }
        self.transposition_table = {}  # Cache for board states
        self.game_tree = []  # Store game tree nodes
        self.current_node_id = 0  # ID counter for game tree nodes

    def evaluate_window(self, window, disc):
        """Evaluate a 4-piece window for scoring"""
        opp_disc = 1 if disc == 2 else 2
        
        # Convert to list if needed
        if hasattr(window, 'tolist'):  # For numpy arrays
            window = window.tolist()
        elif not isinstance(window, list):
            window = list(window)
            
        empty = window.count(0)
        my_pieces = window.count(disc)
        opp_pieces = window.count(opp_disc)
        
        if my_pieces == 4:
            return 1000
        if opp_pieces == 4:
            return -1000
            
        score = 0
        if opp_pieces == 3 and empty == 1:
            score -= 500
        elif my_pieces == 3 and empty == 1:
            score += 100
        elif my_pieces == 2 and empty == 2:
            score += 10
        elif my_pieces == 1 and empty == 3:
            score += 1
            
        return score

    def evaluate_board(self, board):
        """Evaluate the entire board state"""
        score = 0
        disc = self.symbol
        opp_disc = 1 if disc == 2 else 2
        
        # Center column preference
        center_col = len(board[0]) // 2
        center_array = [board[r][center_col] for r in range(len(board))]
        score += center_array.count(disc) * 6
        
        # Horizontal windows
        for r in range(len(board)):
            for c in range(len(board[0]) - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self.evaluate_window(window, disc)
                
        # Vertical windows
        for c in range(len(board[0])):
            for r in range(len(board) - 3):
                window = [board[r+i][c] for i in range(4)]
                score += self.evaluate_window(window, disc)
                
        # Diagonal / windows
        for r in range(len(board) - 3):
            for c in range(len(board[0]) - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, disc)
                
        # Diagonal \ windows
        for r in range(3, len(board)):
            for c in range(len(board[0]) - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, disc)
                
        return score

    def select_move(self, board, disc):
        """Select move for given disc"""
        self.symbol = disc  # Set current player
        return self.get_move(board)  # Get best move

    def get_move_metrics(self, board):
        """Get move with performance metrics"""
        # Reset metrics for new move
        self.metrics['nodes_expanded'] = 0
        self.metrics['max_depth_reached'] = 0
        self.metrics['pruning_counts'] = 0
        
        # Start performance tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Run minimax algorithm
        score, best_col = self.minimax(board, self.max_depth, -math.inf, math.inf, True)
        
        # Calculate performance metrics
        elapsed = time.time() - start_time
        mem_usage = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # Convert to MB
        tracemalloc.stop()
        
        # Package metrics
        metrics = {
            'nodes': self.metrics['nodes_expanded'],
            'depth': self.metrics['max_depth_reached'],
            'time': elapsed,
            'memory': mem_usage,
            'pruning': self.metrics['pruning_counts']
        }
        
        return best_col, metrics

    def get_move(self, board):
        """Public method to get the agent's move with iterative deepening"""
        # Reset metrics and transposition table
        self.metrics['nodes_expanded'] = 0
        self.metrics['max_depth_reached'] = 0
        self.metrics['pruning_counts'] = 0
        self.transposition_table.clear()
        self.game_tree = []  # Reset game tree
        self.current_node_id = 0  # Reset node counter
        
        # Start performance tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Initialize search variables
        best_col = None
        best_score = -math.inf
        valid_cols = [c for c in range(COLS) if board[0][c] == 0]
        
        if not valid_cols:  # No valid moves
            return None

        # Iterative deepening search
        for depth in range(1, self.max_depth + 1):
            try:
                # Run minimax with current depth
                score, col = self.minimax(
                    board, 
                    depth, 
                    -math.inf, 
                    math.inf, 
                    True,
                    start_time,
                    self.time_limit
                )
                
                # Update best move if improved
                if score > best_score:
                    best_score = score
                    best_col = col
                    
                self.metrics['iterative_deepening_depths'].append(depth)
                
                # Early termination if winning move found
                if best_score >= 1000:
                    break
                    
            except TimeoutError:  # Time limit exceeded
                break
        
        # Calculate final metrics
        elapsed = time.time() - start_time
        mem_usage = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
        tracemalloc.stop()
        
        # Store metrics
        self.metrics['execution_times'].append(elapsed)
        self.metrics['memory_usages'].append(mem_usage)
        
        # Visualize game tree
        self.visualize_game_tree()
        
        # Return best move or random if none found
        return best_col if best_col is not None else random.choice(valid_cols)

    def minimax(self, board, depth, alpha, beta, maximizing, start_time=None, time_limit=None, parent_id=None):
        """Minimax algorithm with alpha-beta pruning"""
        # Check time limit
        if time_limit and start_time and (time.time() - start_time) > time_limit:
            raise TimeoutError("Time limit exceeded")
            
        # Create node info for game tree visualization
        node_id = self.current_node_id
        self.current_node_id += 1
        
        node_info = {
            'id': node_id,
            'parent': parent_id,
            'depth': self.max_depth - depth,
            'alpha': alpha,
            'beta': beta,
            'maximizing': maximizing,
            'score': None,
            'move': None,
            'pruned': False
        }
        self.game_tree.append(node_info)

        # Check transposition table for cached results
        board_key = str(board)
        if board_key in self.transposition_table:
            stored_depth, stored_score, stored_col = self.transposition_table[board_key]
            if stored_depth >= depth:
                node_info['score'] = stored_score
                node_info['move'] = stored_col
                return stored_score, stored_col
            
        # Update metrics
        self.metrics['nodes_expanded'] += 1
        self.metrics['max_depth_reached'] = max(self.metrics['max_depth_reached'], self.max_depth - depth)
        
        # Get valid moves and check terminal state
        valid_cols = [c for c in range(COLS) if board[0][c] == 0]
        is_terminal = self.check_terminal(board)
        
        # Base case: terminal node or depth limit reached
        if depth == 0 or is_terminal:
            score = self.evaluate_board(board)
            self.transposition_table[board_key] = (depth, score, None)
            node_info['score'] = score
            return score, random.choice(valid_cols) if valid_cols else None

        # Sort columns by centrality (better move ordering improves pruning)
        valid_cols.sort(key=lambda x: abs(x - COLS//2))
        
        if maximizing:  # Maximizing player (current agent)
            value = -math.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                # Simulate move
                row = self.get_next_open_row(board, col)
                board_copy = [row[:] for row in board]
                board_copy[row][col] = self.symbol
                
                # Recursive minimax call
                new_score, _ = self.minimax(
                    board_copy, 
                    depth - 1, 
                    alpha, 
                    beta, 
                    False,
                    start_time,
                    time_limit,
                    node_id
                )
                
                # Update node info
                node_info['move'] = col
                node_info['score'] = new_score
                
                # Update best move
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.metrics['pruning_counts'] += 1
                    node_info['pruned'] = True
                    break
                    
            # Cache result
            self.transposition_table[board_key] = (depth, value, best_col)
            node_info['score'] = value
            return value, best_col
        else:  # Minimizing player (opponent)
            value = math.inf
            best_col = valid_cols[0]
            opponent = 1 if self.symbol == 2 else 2
            
            for col in valid_cols:
                # Simulate opponent move
                row = self.get_next_open_row(board, col)
                board_copy = [row[:] for row in board]
                board_copy[row][col] = opponent
                
                # Recursive minimax call
                new_score, _ = self.minimax(
                    board_copy, 
                    depth - 1, 
                    alpha, 
                    beta, 
                    True,
                    start_time,
                    time_limit,
                    node_id
                )
                
                # Update node info
                node_info['move'] = col
                node_info['score'] = new_score
                
                # Update best move
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    self.metrics['pruning_counts'] += 1
                    node_info['pruned'] = True
                    break
                    
            # Cache result
            self.transposition_table[board_key] = (depth, value, best_col)
            node_info['score'] = value
            return value, best_col

    def visualize_game_tree(self, max_depth=3, filename="game_tree.png"):
        #Visualize game tree using matplotlib
        if not self.game_tree:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Organize nodes by depth
        depth_nodes = defaultdict(list)
        for node in self.game_tree:
            if node['depth'] <= max_depth:
                depth_nodes[node['depth']].append(node)
        
        #calculate positions
        pos = {}
        for depth, nodes in depth_nodes.items():
            y = max_depth - depth
            x_spacing = 1.0 / (len(nodes) + 1)
            for i, node in enumerate(nodes): #(https://stackoverflow.com/questions/67730079/how-to-write-a-enumerate-loop)
                pos[node['id']] = (x_spacing * (i + 1), y)
        
        #draw edges
        for node in self.game_tree:
            if node['depth'] > max_depth:
                continue
            if node['parent'] is not None and node['parent'] in pos:
                start = pos[node['parent']]
                end = pos[node['id']]
                plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.3)
                
                #add move label
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                if node['move'] is not None:
                    plt.text(mid_x, mid_y, str(node['move']), 
                            ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # Draw nodes
        for node_id, (x, y) in pos.items():
            node = next(n for n in self.game_tree if n['id'] == node_id)
            color = 'red' if node['maximizing'] else 'blue'
            shape = 'o' if not node['pruned'] else 'x'
            label = f"{node['score']:.1f}" if node['score'] is not None else "?"
            
            plt.plot(x, y, marker=shape, markersize=15, 
                    color=color, markeredgewidth=2)
            plt.text(x, y, label, ha='center', va='center')
        
        plt.title("Minimax Game Tree (Red=Max, Blue=Min)")
        plt.axis('off')
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{filename}')
        plt.close()
        print(f"Game tree visualization saved to results/{filename}")

    def get_next_open_row(self, board, col):
        """Find next empty row in a column"""
        for r in range(ROWS - 1, -1, -1):
            if board[r][col] == 0:
                return r
        return -1  # Column is full

    def check_terminal(self, board):
        """Check if board is in terminal state (win/loss/draw)"""
        if self.check_win(board, self.symbol):
            return True
        opponent = 1 if self.symbol == 2 else 2
        if self.check_win(board, opponent):
            return True
        return all(board[0][c] != 0 for c in range(COLS))

    def check_win(self, board, disc):
        """Check if given disc has a winning position"""
        # Horizontal check
        for c in range(COLS - 3):
            for r in range(ROWS):
                if all(board[r][c + i] == disc for i in range(4)):
                    return True
        # Vertical check
        for c in range(COLS):
            for r in range(ROWS - 3):
                if all(board[r + i][c] == disc for i in range(4)):
                    return True
        # Diagonal / check
        for c in range(COLS - 3):
            for r in range(ROWS - 3):
                if all(board[r + i][c + i] == disc for i in range(4)):
                    return True
        # Diagonal \ check
        for c in range(COLS - 3):
            for r in range(3, ROWS):
                if all(board[r - i][c + i] == disc for i in range(4)):
                    return True
        return False


class MLAgent:
    """
    Machine Learning agent using a pre-trained model for Connect Four.
    References:
    - Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
    - Géron, A. (2019). "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"
    """
    def __init__(self, symbol=1, model_path="models/ml_model.joblib"):
        """
        Initialize ML agent with:
        - Player symbol (default 1)
        - Path to pre-trained model
        """
        self.symbol = symbol  # Player's disc
        self.name = "MLAgent"  # Agent name
        # Performance metrics
        self.metrics = {
            'execution_times': [],  # Move calculation times
            'memory_usages': [],  # Memory usage
            'nodes_expanded': 0,  # Positions evaluated
            'wins': 0,  # Game outcomes
            'draws': 0,
            'losses': 0
        }

        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please train the model first.")
        
        # Load pre-trained model
        self.model = joblib.load(model_path)
        self.feature_columns = list(range(42))  # 6x7 board positions

    def board_to_features(self, board):
        """
        Convert game board to feature vector matching training data format
        Args:
            board: Current game board
        Returns:
            list: Feature vector for model prediction
        """
        feature_vector = []
        for row in board:
            for cell in row:
                if cell == 0:  #empty
                    feature_vector.append(0)
                elif cell == 1:  # player 1
                    feature_vector.append(1)
                elif cell == 2:  # player 2
                    feature_vector.append(-1)
        return feature_vector

    def valid_moves(self, board):
        """
        Get list of valid columns for moves
        Args:
            board: Current game board
        Returns:
            list: Valid column indices
        """
        return [c for c in range(COLS) if board[0][c] == 0]

    def predict_best_move(self, board, player):
        """
        Evaluate moves using ML model to find best predicted move
        Args:
            board: Current game board
            player: Current player (1 or 2)
        Returns:
            int: Best column index
        """
        best_score = -float('inf')
        best_move = None
        valid_cols = self.valid_moves(board)
        
        if not valid_cols:  # No valid moves
            return None

        # Evaluate each valid move
        for col in valid_cols:
            # Create board copy with simulated move
            temp_board = np.copy(board)
            for row in range(ROWS-1, -1, -1):  # Find first empty row
                if temp_board[row][col] == 0:
                    temp_board[row][col] = player
                    break
            
            # Convert to features and predict
            features = self.board_to_features(temp_board)
            
            try:
                # Handle different model types
                if hasattr(self.model, 'predict_proba'):  #probability models
                    proba = self.model.predict_proba([features])[0]
                    # Get winning probability
                    if 1 in self.model.classes_:
                        win_index = list(self.model.classes_).index(1)
                        score = proba[win_index]
                    else:
                        score = 0
                else:  # Regression models
                    score = self.model.predict([features])[0]
                
                # Update best move
                if score > best_score:
                    best_score = score
                    best_move = col
            except Exception as e:
                print(f"Prediction error: {e}")
                continue
        
        # Fallback to random if no good move found
        return best_move if best_move is not None else random.choice(valid_cols)

    def select_move(self, board, player):
        """
        Select move with performance tracking
        Args:
            board: Current game board
            player: Current player (1 or 2)
        Returns:
            dict: Move and performance metrics
        """
        # Start performance tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Get best move
        move = self.predict_best_move(board, player)
        
        # Calculate metrics
        elapsed = time.time() - start_time
        mem_usage = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
        tracemalloc.stop()
        
        # Update metrics
        self.metrics['execution_times'].append(elapsed)
        self.metrics['memory_usages'].append(mem_usage)
        self.metrics['nodes_expanded'] += len(self.valid_moves(board))
        
        # Return move and metrics
        return {
            'move': move,
            'metrics': {
                'execution_time': elapsed,
                'memory_usage': mem_usage,
                'nodes_expanded': len(self.valid_moves(board))
            }
        }

    def update_metrics(self, result):
        """
        Update game outcome metrics
        Args:
            result: Game outcome ("win", "draw", or "loss")
        """
        if result == "win":
            self.metrics["wins"] += 1
        elif result == "draw":
            self.metrics["draws"] += 1
        elif result == "loss":
            self.metrics["losses"] += 1


def get_agent_by_choice(choice, symbol=None):
    """
    Factory function to create agent based on user choice
    Args:
        choice: Integer representing agent type
        symbol: Optional player symbol (1 or 2)
    Returns:
        Agent object or None if invalid choice
    """
    if choice == 1:  # Random agent
        return RandomAgent(symbol)
    elif choice == 2:  # Rule-based agent
        return SmartAgent(symbol)
    elif choice == 3:  # Minimax agent
        return MinimaxAgent(symbol, max_depth=4)
    elif choice == 4:  # Machine Learning agent
        return MLAgent(symbol)
    else:  # Invalid choice
        return None