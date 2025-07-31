from connect4_agents import SmartAgent, MinimaxAgent
from performance_utils import PerformanceUtils
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class SmartVsMinimaxEvaluator:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.reset_stats()
        
    def reset_stats(self):
        self.stats = {
            'smart_vs_minimax': {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'game_lengths': [],
                'win_patterns': defaultdict(int),
                'win_rate': 0.0,
                'avg_game_length': 0.0
            },
            'minimax_metrics': {
                'nodes_expanded': [],
                'search_depth': [],
                'branching_factors': [],
                'execution_time': [],
                'memory_usage': [],
                'pruning_counts': 0,
                'pruned_nodes': [],
                'avg_nodes': 0.0,
                'avg_time': 0.0,
                'avg_memory': 0.0,
                'pruning_rate': 0.0
            },
            'resource_metrics': {
                'total_time': 0,
                'peak_memory': 0.0,
                'time_per_game': 0.0
            },
            'smart_metrics': {
                'decision_types': {'win': 0, 'block': 0, 'random': 0},
                'execution_times': [],
                'avg_decision_time': 0.0
            }
        }

    def _record_minimax_metrics(self, metrics):
        """Records Minimax performance metrics for analysis."""
        try:
            self.stats['minimax_metrics']['nodes_expanded'].append(metrics['nodes'])
            self.stats['minimax_metrics']['search_depth'].append(metrics['depth'])
            self.stats['minimax_metrics']['execution_time'].append(metrics['time'])
            self.stats['minimax_metrics']['memory_usage'].append(metrics['memory'])
            self.stats['minimax_metrics']['pruning_counts'] += metrics['pruning']
            self.stats['minimax_metrics']['pruned_nodes'].append(metrics['pruning'])
            
            # Calculate branching factor for this move
            if metrics['depth'] > 0:
                branching_factor = metrics['nodes'] ** (1/metrics['depth'])
                self.stats['minimax_metrics']['branching_factors'].append(branching_factor)
        except KeyError as e:
            print(f"Warning: Missing metric {e} in minimax metrics")


    def _record_result(self, matchup_name, winner_agent, smart_agent, game_length, board):
        """Records the result of a completed game."""
        stats = self.stats[matchup_name]
        if winner_agent == smart_agent:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        stats['game_lengths'].append(game_length)
        
        if PerformanceUtils.check_for_win(board, winner_agent.symbol):
            win_pattern = self._detect_win_pattern(board, winner_agent.symbol)
            stats['win_patterns'][win_pattern] += 1

    def _detect_win_pattern(self, board, symbol):
    # Horizontal check
        for row in range(PerformanceUtils.ROWS):
            for col in range(PerformanceUtils.COLS - 3):
                if all(board[row][col + i] == symbol for i in range(4)):
                    return 'Horizontal'
    # Vertical check
        for row in range(PerformanceUtils.ROWS - 3):
            for col in range(PerformanceUtils.COLS):
                if all(board[row + i][col] == symbol for i in range(4)):
                    return 'Vertical'
    # Diagonal checks
        for row in range(PerformanceUtils.ROWS - 3):
            for col in range(PerformanceUtils.COLS - 3):
                if all(board[row + i][col + i] == symbol for i in range(4)):
                    return 'Diagonal (\\)'  

        for row in range(3, PerformanceUtils.ROWS):
            for col in range(PerformanceUtils.COLS - 3):
                if all(board[row - i][col + i] == symbol for i in range(4)):
                    return 'Diagonal (/)'
                return 'Unknown'

    def _collect_full_metrics(self, start_time, num_games):
        """Calculates final performance metrics with proper checks."""
        # Get peak memory before stopping tracemalloc
        peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # Convert to MB
        
        # Resource metrics
        self.stats['resource_metrics']['total_time'] = time.time() - start_time
        self.stats['resource_metrics']['time_per_game'] = (
            self.stats['resource_metrics']['total_time'] / num_games
        )
        self.stats['resource_metrics']['peak_memory'] = float(peak_mem)
        
        # Game statistics
        stats = self.stats['smart_vs_minimax']
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        if total_games > 0:
            stats['win_rate'] = stats['wins'] / total_games * 100
            if stats['game_lengths']:  # Check if not empty
                stats['avg_game_length'] = np.mean(stats['game_lengths'])
        
        # Minimax metrics
        mm = self.stats['minimax_metrics']
        if mm['nodes_expanded']:
            mm['avg_nodes'] = np.mean(mm['nodes_expanded']) if mm['nodes_expanded'] else 0
            mm['avg_time'] = np.mean(mm['execution_time']) if mm['execution_time'] else 0
            mm['avg_memory'] = np.mean(mm['memory_usage']) if mm['memory_usage'] else 0
            
            # Safe pruning rate calculation
            total_nodes = sum(mm['nodes_expanded'])
            if total_nodes > 0:
                mm['pruning_rate'] = (sum(mm['pruned_nodes']) / total_nodes) * 100
        
        # Smart metrics
        smart = self.stats['smart_metrics']
        if smart['execution_times']:
            smart['avg_decision_time'] = np.mean(smart['execution_times'])

    def evaluate(self, num_games=100):
        self.reset_stats()
        smart_agent = SmartAgent(symbol=1)
        minimax_agent = MinimaxAgent(symbol=2, max_depth=self.max_depth)
        
        print(f"Evaluating Smart vs Minimax Agent ({num_games} games)...")
        start_time = time.time()
        tracemalloc.start()
        
        for game_num in range(num_games):
            if game_num % 10 == 0:
                print(f"  Completed {game_num}/{num_games} games...")
            
            board = PerformanceUtils.create_new_board()
            game_length = 0
            current_agent = smart_agent
            
            while True:
                move_start = time.time()
                
                if current_agent == minimax_agent:
                    move, metrics = minimax_agent.get_move_metrics(board)
                    self._record_minimax_metrics(metrics)
                    move_time = time.time() - move_start
                    self.stats['minimax_metrics']['execution_time'].append(move_time)

                    # Get current memory usage in MB / (https://stackoverflow.com/questions/70525623/measuring-the-allocated-memory-with-tracemalloc)
                    current_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
                    self.stats['minimax_metrics']['memory_usage'].append(current_memory)
                else:
                    move = current_agent.get_move(board)
                    decision_type = self._infer_decision_type(board, move, current_agent.symbol)
                    self.stats['smart_metrics']['decision_types'][decision_type] += 1
                    self.stats['smart_metrics']['execution_times'].append(time.time() - move_start)
                
                PerformanceUtils.make_move(board, move, current_agent.symbol)
                game_length += 1
                
                if PerformanceUtils.check_for_win(board, current_agent.symbol):
                    self._record_result('smart_vs_minimax', current_agent, smart_agent, game_length, board)
                    break
                if PerformanceUtils.is_draw(board):
                    self.stats['smart_vs_minimax']['draws'] += 1
                    self.stats['smart_vs_minimax']['game_lengths'].append(game_length)
                    break
                    
                current_agent = minimax_agent if current_agent == smart_agent else smart_agent
        
        self._collect_full_metrics(start_time, num_games)
        tracemalloc.stop()
        PerformanceUtils.save_results('results/smart_vs_minimax_results.csv', self.stats, 'smart_vs_minimax')
        self._generate_performance_report()
        self._visualize_advanced_metrics()
        return self.stats

    def _infer_decision_type(self, board, move, symbol):
        """Determine if move was a win, block, or random choice."""
        temp_board = [row[:] for row in board]
        PerformanceUtils.make_move(temp_board, move, symbol)
        if PerformanceUtils.check_for_win(temp_board, symbol):
            return 'win'
        
        opponent_symbol = 3 - symbol
        for col in range(PerformanceUtils.COLS):
            temp_board = [row[:] for row in board]
            if PerformanceUtils.make_move(temp_board, col, opponent_symbol):
                if PerformanceUtils.check_for_win(temp_board, opponent_symbol):
                    if col == move:
                        return 'block'
        return 'random'


    #performance evaluation 
    def _generate_performance_report(self):
        """Generate comprehensive performance report with proper checks."""
        print("\n=== SMART vs MINIMAX PERFORMANCE REPORT ===")
        
        # 1. Accuracy Metrics
        stats = self.stats['smart_vs_minimax']
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        print(f"\n1. Accuracy Metrics ({total_games} games):")
        print(f"  - Smart Win Rate: {stats['win_rate']:.1f}%")
        print(f"  - Minimax Win Rate: {100 - stats['win_rate']:.1f}%")
        print(f"  - Draw Rate: {stats['draws']/total_games*100:.1f}%" if total_games > 0 else "  - Draw Rate: 0.0%")
        
        # 2. Game-Level Metrics
        print(f"\n2. Game-Level Metrics:")
        print(f"  - Average Game Length: {stats['avg_game_length']:.1f} moves" if stats['game_lengths'] else "  - Average Game Length: N/A")
        
        # Format win patterns nicely
        win_patterns = {k: v for k, v in stats['win_patterns'].items() if v > 0}
        print(f"  - Win Patterns: {win_patterns or 'None'}")
        
        # 3. Smart Agent Decision Making
        print(f"\n3. Smart Agent Decision Making:")
        decisions = self.stats['smart_metrics']['decision_types']
        total = sum(decisions.values())
        if total > 0:
            print(f"  - Winning Moves: {decisions['win']/total*100:.1f}%")
            print(f"  - Blocking Moves: {decisions['block']/total*100:.1f}%")
            print(f"  - Random Moves: {decisions['random']/total*100:.1f}%")
        else:
            print("  - No decisions recorded")
        
        # 4. Minimax Search Metrics
        print(f"\n4. Minimax Search Metrics:")
        mm = self.stats['minimax_metrics']
        print(f"  - Avg Nodes Expanded: {mm['avg_nodes']:.0f}" if mm['nodes_expanded'] else "  - Avg Nodes Expanded: N/A")
        print(f"  - Avg Search Depth: {np.mean(mm['search_depth']):.1f}" if mm['search_depth'] else "  - Avg Search Depth: N/A")
        print(f"  - Avg Branching Factor: {np.mean(mm['branching_factors']):.2f}" if mm['branching_factors'] else "  - Avg Branching Factor: N/A")
        print(f"  - Pruning Rate: {mm['pruning_rate']:.1f}%" if mm['nodes_expanded'] else "  - Pruning Rate: N/A")
        
        # 5. Efficiency Metrics
        print(f"\n5. Efficiency Metrics:")
        res = self.stats['resource_metrics']
        print(f"  - Total Time: {res['total_time']:.2f}s")
        print(f"  - Time per Game: {res['time_per_game']:.2f}s")
        print(f"  - Peak Memory: {res['peak_memory']:.2f} MB")
        print(f"  - Avg Minimax Time: {mm['avg_time']:.4f}s" if mm['execution_time'] else "  - Avg Minimax Time: N/A")
        print(f"  - Avg Smart Time: {self.stats['smart_metrics']['avg_decision_time']:.4f}s" 
              if self.stats['smart_metrics']['execution_times'] else "  - Avg Smart Time: N/A")


    #This method creates a set of histograms to visualize key performance metrics for the Minimax algorithm, including:
    #1. Nodes Expanded: The number of nodes the algorithm explores during its search.
    #2. Search Depth: The maximum depth the algorithm reaches in its search tree.
    #3. Branching Factors: The average number of child nodes each node in the search tree generates.
    #4. Execution Time: The time taken per move for the algorithm to compute its next move.
    #5. Memory Usage: The amount of memory used by the algorithm during its search process.

    #Each metric is represented by a histogram, and if no data is available for a metric, a "No Data" message is displayed. This helps in understanding the computational behavior and efficiency of the Minimax algorithm
    
    def _visualize_advanced_metrics(self):
        """Generate all visualizations for Minimax performance."""
        mm = self.stats['minimax_metrics']
        plt.figure(figsize=(15, 10))
        
        # 1. Nodes Expanded
        plt.subplot(2, 3, 1)
        plt.hist(mm['nodes_expanded'], bins=20, color='blue') if mm['nodes_expanded'] else plt.text(0.5, 0.5, 'No Data', ha='center')
        plt.title('Nodes Expanded Distribution')
        plt.xlabel('Nodes')
        
        # 2. Search Depth
        plt.subplot(2, 3, 2)
        if mm['search_depth']:
            plt.hist(mm['search_depth'], bins=range(1, self.max_depth+2), color='green')
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center')
        plt.title('Search Depth Distribution')
        plt.xlabel('Depth')
        
        # 3. Branching Factors
        plt.subplot(2, 3, 3)
        plt.hist(mm['branching_factors'], bins=20, color='orange') if mm['branching_factors'] else plt.text(0.5, 0.5, 'No Data', ha='center')
        plt.title('Branching Factor Distribution')
        plt.xlabel('Branching Factor')
        
        # 4. Execution Time
        plt.subplot(2, 3, 4)
        plt.hist(mm['execution_time'], bins=20, color='red') if mm['execution_time'] else plt.text(0.5, 0.5, 'No Data', ha='center')
        plt.title('Execution Time per Move (s)')
        plt.xlabel('Seconds')
        
        # 5. Memory Usage
        plt.subplot(2, 3, 5)
        plt.hist(mm['memory_usage'], bins=20, color='purple') if mm['memory_usage'] else plt.text(0.5, 0.5, 'No Data', ha='center')
        plt.title('Memory Usage per Move (MB)')
        plt.xlabel('MB')
        
        # 6. Pruning Rates (https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
        plt.subplot(2, 3, 6)
        if mm['nodes_expanded'] and mm['pruned_nodes']:
            pruning_rates = [p/(p+n) if (p+n) > 0 else 0 
                            for p, n in zip(mm['pruned_nodes'], mm['nodes_expanded'])]
            plt.hist(pruning_rates, bins=20, color='brown')
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center')
        plt.title('Pruning Rate Distribution')
        plt.xlabel('Pruning Rate')
        
        plt.tight_layout()
        plt.savefig('results/minimax_performance.png') #saving it into the results folder
        plt.show()