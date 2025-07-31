# Importing required agent classes for simulation
from connect4_agents import RandomAgent, SmartAgent  # Importing Random and Smart agents for the simulation

# Importing libraries for performance tracking and visualizations
import time  # For measuring execution time
import tracemalloc  # For tracking memory usage (https://docs.python.org/3/library/tracemalloc.html)
import matplotlib.pyplot as plt  # For generating performance charts
import numpy as np  # For numerical calculations like averages
from collections import defaultdict  # For storing win pattern frequencies efficiently

# Importing utility class containing helper methods
from performance_utils import PerformanceUtils 

# Class for evaluating performance of Random Agent vs Smart Agent
class RandomVsSmartEvaluator:
    def __init__(self):
        self.reset_stats()  # Initialises statistics

    # Method to reset all stats before starting evaluation
    def reset_stats(self):
        self.stats = {
            'random_vs_smart': {
                'wins': 0,          # Smart Agent wins
                'losses': 0,        # Random Agent wins
                'draws': 0,         # Draws
                'game_lengths': [], # Length of each game
                'win_rate': 0       # Smart Agent win rate
            },
            'resource_metrics': {   # Tracks time and memory
                'total_time': 0,
                'peak_memory': 0
            },
            'smart_metrics': {      # Specific stats for Smart Agent
                'decision_types': {'win': 0, 'block': 0, 'random': 0},
                'execution_times': [],
                'nodes_expanded': 0
            },
            'win_patterns': defaultdict(int)  # Track patterns of winning moves
        }

    # Main method to run simulations between Random and Smart agents
    def evaluate(self, num_games=500):  # Default to 500 games
        self.reset_stats()
        random_agent = RandomAgent(symbol=1)  # Symbol 1 = Random agent
        smart_agent = SmartAgent(symbol=2)    # Symbol 2 = Smart agent
        
        print(f"Evaluating Random vs Smart Agent ({num_games} games)...")
        start_time = time.time()  # Start timer
        tracemalloc.start()  # Start memory tracking
        
        for game_num in range(num_games):  # Loop through each game
            board = PerformanceUtils.create_new_board()
            game_length = 0
            current_agent = random_agent  # Random agent starts first
            
            while True:  # Game loop
                if current_agent == smart_agent:
                    move_start = time.time()
                    move = smart_agent.get_move(board)  # Smart agent decides move
                    decision_type = self._infer_decision_type(board, move, smart_agent.symbol)
                    self.stats['smart_metrics']['decision_types'][decision_type] += 1
                    self.stats['smart_metrics']['execution_times'].append(time.time() - move_start)
                else:
                    move = random_agent.get_move(board)  # Random agent move
                
                PerformanceUtils.make_move(board, move, current_agent.symbol)
                game_length += 1
                
                if PerformanceUtils.check_for_win(board, current_agent.symbol):
                    self._record_result(current_agent, random_agent, game_length, board)
                    break  # End game if someone wins
                
                if PerformanceUtils.is_draw(board):
                    self.stats['random_vs_smart']['draws'] += 1
                    self.stats['random_vs_smart']['game_lengths'].append(game_length)
                    break  # End game if it's a draw
                
                # Switch players
                current_agent = smart_agent if current_agent == random_agent else random_agent
        
        self._finalize_metrics(start_time)
        self._generate_performance_report()
        return self.stats

    # Helper method to classify Smart Agent's move type
    def _infer_decision_type(self, board, move, symbol):
        """Determine if Smart Agent move was a win, block, or random"""
        if symbol is None:
            return 'random'
        
        # Check if move leads to a win
        temp_board = [row[:] for row in board]
        PerformanceUtils.make_move(temp_board, move, symbol)
        if PerformanceUtils.check_for_win(temp_board, symbol):
            return 'win'
        
        # Check if move blocks opponent's win
        opponent_symbol = 3 - symbol
        for col in range(PerformanceUtils.COLS):
            temp_board = [row[:] for row in board]
            if PerformanceUtils.make_move(temp_board, col, opponent_symbol):
                if PerformanceUtils.check_for_win(temp_board, opponent_symbol):
                    if col == move:
                        return 'block'
        return 'random'

    # Records win/loss result for each game
    def _record_result(self, current_agent, random_agent, game_length, board):
        """Record the result of the completed game"""
        if current_agent == random_agent:
            self.stats['random_vs_smart']['losses'] += 1  # Random Agent won
        else:
            self.stats['random_vs_smart']['wins'] += 1    # Smart Agent won
        
        win_pattern = 'generic'  
        self.stats['win_patterns'][win_pattern] += 1
        self.stats['random_vs_smart']['game_lengths'].append(game_length)

    # Final calculation of metrics after all games
    def _finalize_metrics(self, start_time):
        self.stats['resource_metrics']['total_time'] = time.time() - start_time  #total time taken
        self.stats['resource_metrics']['peak_memory'] = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  #memory in MB
        tracemalloc.stop()
        
        # Calculate win rate for Smart Agent
        total_games = (self.stats['random_vs_smart']['wins'] + 
                      self.stats['random_vs_smart']['losses'] + 
                      self.stats['random_vs_smart']['draws'])
        if total_games > 0:
            self.stats['random_vs_smart']['win_rate'] = (
                self.stats['random_vs_smart']['wins'] / total_games * 100
            )
        
        # Calculate average game length
        if self.stats['random_vs_smart']['game_lengths']:
            self.stats['random_vs_smart']['avg_game_length'] = np.mean(
                self.stats['random_vs_smart']['game_lengths']
            )
        
        # Calculate average decision time
        if self.stats['smart_metrics']['execution_times']:
            self.stats['smart_metrics']['avg_decision_time'] = np.mean(
                self.stats['smart_metrics']['execution_times']
            )

    #generates printed report of the evaluation
    def _generate_performance_report(self):
        print("\n=== SMART AGENT vs RANDOM AGENT PERFORMANCE REPORT ===")
        print(f"\n1. Game Outcomes ({len(self.stats['random_vs_smart']['game_lengths'])} games):")
        print(f"  - Smart Agent Wins: {self.stats['random_vs_smart']['wins']} "
              f"({self.stats['random_vs_smart']['win_rate']:.1f}%)")
        print(f"  - Random Agent Wins: {self.stats['random_vs_smart']['losses']} "
              f"({100 - self.stats['random_vs_smart']['win_rate']:.1f}%)")
        print(f"  - Draws: {self.stats['random_vs_smart']['draws']} "
              f"({self.stats['random_vs_smart']['draws']/len(self.stats['random_vs_smart']['game_lengths'])*100:.1f}%)")
        
        print(f"\n2. Game-Level Metrics:")
        print(f"  - Average Game Length: {self.stats['random_vs_smart']['avg_game_length']:.1f} moves")
        
        print(f"\n3. Smart Agent Decision Making:")
        decisions = self.stats['smart_metrics']['decision_types']
        total = sum(decisions.values())
        if total > 0:
            print(f"  - Winning Moves: {decisions['win']} ({decisions['win']/total*100:.1f}%)")
            print(f"  - Blocking Moves: {decisions['block']} ({decisions['block']/total*100:.1f}%)")
            print(f"  - Random Moves: {decisions['random']} ({decisions['random']/total*100:.1f}%)")
        else:
            print("  - No smart agent decisions were recorded.")
        
        print(f"\n4. Efficiency Metrics:")
        print(f"  - Total Evaluation Time: {self.stats['resource_metrics']['total_time']:.2f}s")
        print(f"  - Avg Decision Time: {self.stats['smart_metrics']['avg_decision_time']:.6f}s")
        print(f"  - Peak Memory Usage: {self.stats['resource_metrics']['peak_memory']:.2f} MB")
        
        self._visualise_performance()  # Show graphs

    # Method to visualize the results with matplotlib charts
    def _visualise_performance(self):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

        # Pie chart of win/loss/draw
        axs[0, 0].pie(
            [self.stats['random_vs_smart']['wins'], 
             self.stats['random_vs_smart']['losses'], 
             self.stats['random_vs_smart']['draws']],
            labels=['Smart Wins', 'Random Wins', 'Draws'],
            autopct='%1.1f%%',
            colors=['green', 'red', 'blue'],
            textprops={'fontsize': 10}
        )
        axs[0, 0].set_title('Win/Loss/Draw Distribution', fontsize=12)

        # Bar chart for smart agent decision types
        decisions = self.stats['smart_metrics']['decision_types']
        axs[0, 1].bar(decisions.keys(), decisions.values(), width=0.6, edgecolor='black')
        axs[0, 1].set_title('Smart Agent Decision Types', fontsize=12)
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].tick_params(axis='x', labelrotation=20, labelsize=10)
        axs[0, 1].tick_params(axis='y', labelsize=10)

        #histogram of Smart Agent decision times
        axs[1, 0].hist(self.stats['smart_metrics']['execution_times'], bins=20, edgecolor='black')
        axs[1, 0].set_title('Decision Time Distribution', fontsize=12)
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].tick_params(axis='both', labelsize=10)

        #histogram of game lengths
        axs[1, 1].hist(self.stats['random_vs_smart']['game_lengths'], bins=20, color='purple', edgecolor='black')
        axs[1, 1].set_title('Game Length Distribution', fontsize=12)
        axs[1, 1].set_xlabel('Moves')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].tick_params(axis='both', labelsize=10)

        # Save the visual as an image
        fig.savefig('results/random_vs_smart_performance.png')

        # Display the plots
        plt.show()