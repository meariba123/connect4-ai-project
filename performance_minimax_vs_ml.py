import numpy as np
import time
import tracemalloc
import gc
from collections import defaultdict
from connect4_agents import MLAgent, MinimaxAgent
import matplotlib.pyplot as plt
import os
from performance_utils import PerformanceUtils


class MinimaxVsMLEvaluator:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model_path = model_path
        self.minimax_agent = MinimaxAgent(symbol=1, max_depth=4)
        self.ml_agent = MLAgent(symbol=2, model_path=model_path)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'ml_vs_minimax': {
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
            'ml_metrics': {
                'execution_times': [],
                'memory_usages': [],
                'avg_confidence': 0.0,
                'confidences': []
            },
            'resource_metrics': {
                'total_time': 0,
                'peak_memory': 0.0,
                'time_per_game': 0.0
            }
        }

    def evaluate(self, num_games=50, batch_size=50): #batches the games so it doesnt take as much memory and makes it easier to understand 
        self.reset_stats()
        batches = num_games // batch_size
        remaining = num_games % batch_size

        print("Starting Minimax vs ML evaluation...")
        start_time = time.time()
        tracemalloc.start()

        for batch in range(batches + (1 if remaining else 0)):
            current_batch = batch_size if batch < batches else remaining
            if current_batch == 0:
                continue

            print(f"\nProcessing batch {batch+1}/{(batches + (1 if remaining else 0))} ({current_batch} games)")
            for game_index in range(current_batch):
                gc.collect()
                if (batch * batch_size + game_index) % 2 == 0:
                    agent1, agent2 = self.minimax_agent, self.ml_agent
                else:
                    agent1, agent2 = self.ml_agent, self.minimax_agent

                winner, moves, metrics = self.play_enhanced_game(agent1, agent2)

                if winner == 1:
                    if isinstance(agent1, MinimaxAgent):
                        self.stats['ml_vs_minimax']['losses'] += 1
                    else:
                        self.stats['ml_vs_minimax']['wins'] += 1
                elif winner == 2:
                    if isinstance(agent2, MinimaxAgent):
                        self.stats['ml_vs_minimax']['losses'] += 1
                    else:
                        self.stats['ml_vs_minimax']['wins'] += 1
                else:
                    self.stats['ml_vs_minimax']['draws'] += 1

                self.stats['ml_vs_minimax']['game_lengths'].append(moves)
                if winner != 0:
                    win_pattern = self.get_win_pattern(agent1.last_board if winner == 1 else agent2.last_board)
                    self.stats['ml_vs_minimax']['win_patterns'][win_pattern] += 1

                if (game_index + 1) % 10 == 0:
                    print(f"Completed {game_index + 1}/{current_batch} games")

        self._collect_full_metrics(start_time, num_games)
        tracemalloc.stop()
        self._generate_performance_report()
        self._visualize_enhanced_metrics()
        return self.stats

    def play_enhanced_game(self, agent1, agent2):
        board = np.zeros((PerformanceUtils.ROWS, PerformanceUtils.COLS), dtype=int)
        agents = {1: agent1, 2: agent2}
        current_player = 1
        move_count = 0
        game_metrics = {'minimax': [], 'ml': []}

        while True:
            move_count += 1
            if isinstance(agents[current_player], MinimaxAgent):
                move, metrics = agents[current_player].get_move_metrics(board)
                self._record_minimax_metrics(metrics)
                game_metrics['minimax'].append(metrics)
            else:
                tracemalloc.start()
                start_time = time.time()
                move_result = agents[current_player].select_move(board, current_player)
                ml_metrics = {
                    'execution_time': time.time() - start_time,
                    'memory_usage': tracemalloc.get_traced_memory()[1] / (1024 * 1024)
                }
                tracemalloc.stop()
                if isinstance(move_result, dict):
                    move = move_result['move']
                    metrics = {**ml_metrics, **move_result.get('metrics', {})}
                else:
                    move = move_result
                    metrics = ml_metrics
                game_metrics['ml'].append(metrics)
                self._record_ml_metrics(metrics)

            row = next((r for r in range(PerformanceUtils.ROWS - 1, -1, -1) if board[r][move] == 0), -1)
            if row == -1:
                return 3 - current_player, move_count, game_metrics

            board[row][move] = current_player
            if self.check_win(board, current_player):
                agents[current_player].last_board = board.copy()
                return current_player, move_count, game_metrics
            if self.check_draw(board):
                return 0, move_count, game_metrics

            current_player = 3 - current_player

    def _record_minimax_metrics(self, metrics):
        try:
            self.stats['minimax_metrics']['nodes_expanded'].append(metrics['nodes'])
            self.stats['minimax_metrics']['search_depth'].append(metrics['depth'])
            self.stats['minimax_metrics']['execution_time'].append(metrics['time'])
            self.stats['minimax_metrics']['memory_usage'].append(metrics['memory'])
            self.stats['minimax_metrics']['pruning_counts'] += metrics['pruning']
            self.stats['minimax_metrics']['pruned_nodes'].append(metrics['pruning'])
            if metrics['depth'] > 0:
                branching_factor = metrics['nodes'] ** (1 / metrics['depth'])
                self.stats['minimax_metrics']['branching_factors'].append(branching_factor)
        except KeyError as e:
            print(f"Warning: Missing metric {e} in minimax metrics")


    # This method records performance metrics specific to the ML agent, including:
    #1. Execution Time: The time taken by the ML model to compute its move.
    #2. Memory Usage: The amount of memory consumed by the ML model during execution.
    #3. Confidence: The confidence level of the ML model's prediction, if available.
    def _record_ml_metrics(self, metrics):
        self.stats['ml_metrics']['execution_times'].append(metrics.get('execution_time', 0))
        self.stats['ml_metrics']['memory_usages'].append(metrics.get('memory_usage', 0))
        if 'confidence' in metrics:
            self.stats['ml_metrics']['confidences'].append(metrics['confidence'])


    #This method aggregates and computes various resource and performance metrics across multiple games, including:
    #1. Total Time: The total execution time for all games.
    #2. Time per Game: The average time taken per game.
    #3. Peak Memory Usage: The maximum memory usage during the execution, tracked with tracemalloc.
    
    #It also computes game-specific metrics for the Minimax and ML agents:
    #- Minimax Metrics: Average nodes expanded, average execution time, average memory usage, and pruning rate.
    #- ML Metrics: Average execution time, average memory usage, and average confidence.
    def _collect_full_metrics(self, start_time, num_games):
        self.stats['resource_metrics']['total_time'] = time.time() - start_time
        self.stats['resource_metrics']['time_per_game'] = (
            self.stats['resource_metrics']['total_time'] / num_games
        )
        self.stats['resource_metrics']['peak_memory'] = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

        stats = self.stats['ml_vs_minimax']
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        if total_games > 0:
            stats['win_rate'] = stats['wins'] / total_games * 100
            if stats['game_lengths']:
                stats['avg_game_length'] = np.mean(stats['game_lengths'])

        mm = self.stats['minimax_metrics']
        if mm['nodes_expanded']:
            mm['avg_nodes'] = np.mean(mm['nodes_expanded'])
            mm['avg_time'] = np.mean(mm['execution_time'])
            mm['avg_memory'] = np.mean(mm['memory_usage'])
            total_nodes = sum(mm['nodes_expanded'])
            if total_nodes > 0:
                mm['pruning_rate'] = (sum(mm['pruned_nodes']) / total_nodes) * 100

        ml = self.stats['ml_metrics']
        if ml['execution_times']:
            ml['avg_time'] = np.mean(ml['execution_times'])
            ml['avg_memory'] = np.mean(ml['memory_usages'])
        if ml['confidences']:
            ml['avg_confidence'] = np.mean(ml['confidences'])

    def check_win(self, board, player):
        for c in range(PerformanceUtils.COLS - 3):
            for r in range(PerformanceUtils.ROWS):
                if (board[r][c] == player and board[r][c+1] == player and
                    board[r][c+2] == player and board[r][c+3] == player):
                    return True
        for c in range(PerformanceUtils.COLS):
            for r in range(PerformanceUtils.ROWS - 3):
                if (board[r][c] == player and board[r+1][c] == player and
                    board[r+2][c] == player and board[r+3][c] == player):
                    return True
        for c in range(PerformanceUtils.COLS - 3):
            for r in range(PerformanceUtils.ROWS - 3):
                if (board[r][c] == player and board[r+1][c+1] == player and
                    board[r+2][c+2] == player and board[r+3][c+3] == player):
                    return True
        for c in range(PerformanceUtils.COLS - 3):
            for r in range(3, PerformanceUtils.ROWS):
                if (board[r][c] == player and board[r-1][c+1] == player and
                    board[r-2][c+2] == player and board[r-3][c+3] == player):
                    return True
        return False

    def check_draw(self, board):
        """Check if the game is a draw"""
        return np.all(board[0] != 0)

    def get_win_pattern(self, board):
        """Identify winning pattern (horizontal/vertical/diagonal)"""
        # Check horizontal wins
        for r in range(PerformanceUtils.ROWS):
            for c in range(PerformanceUtils.COLS - 3):
                if board[r][c] != 0 and board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                    return "horizontal"
        
        # Check vertical wins
        for c in range(PerformanceUtils.COLS):
            for r in range(PerformanceUtils.ROWS - 3):
                if board[r][c] != 0 and board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                    return "vertical"
        
        # Check diagonal (/) wins
        for r in range(PerformanceUtils.ROWS - 3):
            for c in range(PerformanceUtils.COLS - 3):
                if board[r][c] != 0 and board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                    return "diagonal_positive"
        
        # Check diagonal (\) wins
        for r in range(3, PerformanceUtils.ROWS):
            for c in range(PerformanceUtils.COLS - 3):
                if board[r][c] != 0 and board[r][c] == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                    return "diagonal_negative"
        
        return "unknown"

    def _generate_performance_report(self):
        ##Generate comprehensive performance report
        print("\n=== MINIMAX vs ML PERFORMANCE REPORT ===")
        
        # 1. Accuracy Metrics
        stats = self.stats['ml_vs_minimax']
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        print(f"\n1. Accuracy Metrics ({total_games} games):")
        print(f"  - ML Win Rate: {stats['win_rate']:.1f}%")
        print(f"  - Minimax Win Rate: {(stats['losses']/total_games*100):.1f}%" if total_games > 0 else "  - Minimax Win Rate: 0.0%")
        print(f"  - Draw Rate: {(stats['draws']/total_games*100):.1f}%" if total_games > 0 else "  - Draw Rate: 0.0%")
        
        # 2. Game-Level Metrics
        print(f"\n2. Game-Level Metrics:")
        print(f"  - Average Game Length: {stats['avg_game_length']:.1f} moves" if stats['game_lengths'] else "  - Average Game Length: N/A")
        print(f"  - Win Patterns: {dict(stats['win_patterns'])}")
        
        # 3. Minimax Search Metrics
        print(f"\n3. Minimax Search Metrics:")
        mm = self.stats['minimax_metrics']
        print(f"  - Avg Nodes Expanded: {mm['avg_nodes']:.0f}" if mm['nodes_expanded'] else "  - Avg Nodes Expanded: N/A")
        print(f"  - Avg Search Depth: {np.mean(mm['search_depth']):.1f}" if mm['search_depth'] else "  - Avg Search Depth: N/A")
        print(f"  - Avg Branching Factor: {np.mean(mm['branching_factors']):.2f}" if mm['branching_factors'] else "  - Avg Branching Factor: N/A")
        print(f"  - Pruning Rate: {mm['pruning_rate']:.1f}%" if mm['nodes_expanded'] else "  - Pruning Rate: N/A")
        
        # 4. ML Agent Metrics
        print(f"\n4. ML Agent Metrics:")
        ml = self.stats['ml_metrics']

        exec_times = [t for t in ml['execution_times'] if t is not None]
        avg_time = sum(exec_times) / len(exec_times) if exec_times else None
        print(f"  - Avg Inference Time: {avg_time:.4f}s" if avg_time is not None else "  - Avg Inference Time: N/A")

        mem_usages = [m for m in ml['memory_usages'] if m is not None]
        avg_mem = sum(mem_usages) / len(mem_usages) if mem_usages else None
        print(f"  - Avg Memory Usage: {avg_mem:.2f} MB" if avg_mem is not None else "  - Avg Memory Usage: N/A")

        print(f"  - Avg Prediction Confidence: {ml['avg_confidence']:.2f}" if ml['confidences'] else "  - Avg Confidence: N/A")
        
        # 5. Resource Metrics
        print(f"\n5. Resource Metrics:")
        res = self.stats['resource_metrics']
        print(f"  - Total Time: {res['total_time']:.2f}s")
        print(f"  - Time per Game: {res['time_per_game']:.2f}s")
        print(f"  - Peak Memory: {res['peak_memory']:.2f} MB")

    def _visualize_enhanced_metrics(self):
        """Generate enhanced visualizations for both agents"""
        plt.figure(figsize=(18, 12))
        
        # 1.win Rates
        plt.subplot(2, 3, 1)
        stats = self.stats['ml_vs_minimax']
        plt.bar(['ML Agent', 'Minimax', 'Draws'],
                [stats['win_rate'], 
                 (stats['losses']/(stats['wins']+stats['losses']+stats['draws'])*100) if (stats['wins']+stats['losses']+stats['draws']) > 0 else 0,
                 (stats['draws']/(stats['wins']+stats['losses']+stats['draws'])*100) if (stats['wins']+stats['losses']+stats['draws']) > 0 else 0],
                color=['blue', 'red', 'gray'])
        plt.title('Win/Draw Rates')
        plt.ylabel('Percentage (%)')
        
        # 2. Minimax Search Performance
        plt.subplot(2, 3, 2)
        mm = self.stats['minimax_metrics']
        metrics = ['Nodes', 'Depth', 'Branching']
        values = [
            mm['avg_nodes'] if mm['nodes_expanded'] else 0,
            np.mean(mm['search_depth']) if mm['search_depth'] else 0,
            np.mean(mm['branching_factors']) if mm['branching_factors'] else 0
        ]
        plt.bar(metrics, values, color=['green', 'purple', 'orange'])
        plt.title('Minimax Search Performance')
        
        # 3. Pruning Effectiveness
        plt.subplot(2, 3, 3)
        plt.bar(['Pruned Nodes'], 
               [mm['pruning_rate'] if mm['nodes_expanded'] else 0],
               color='cyan')
        plt.title('Pruning Effectiveness')
        plt.ylabel('Percentage (%)')
        
        # 4. Game Length Distribution
        plt.subplot(2, 3, 4)
        if stats['game_lengths']:
            plt.hist(stats['game_lengths'], bins=20, color='skyblue')
        plt.title('Game Length Distribution')
        plt.xlabel('Moves')
        
        # 5. Win Patterns
        plt.subplot(2, 3, 5)
        if stats['win_patterns']:
            patterns, counts = zip(*stats['win_patterns'].items())
            plt.bar(patterns, counts, color='lightgreen')
        plt.title('Winning Patterns')
        plt.xticks(rotation=45)
        
        # 6. Resource Usage Comparison
        plt.subplot(2, 3, 6)
        mm_times = mm['execution_time'] if mm['execution_time'] else [0]
        ml_times = self.stats['ml_metrics']['execution_times'] if self.stats['ml_metrics']['execution_times'] else [0]
        plt.boxplot([mm_times, ml_times], labels=['Minimax', 'ML'])
        plt.title('Move Decision Times')
        plt.ylabel('Seconds')

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/minimax_vs_ml_metrics.png')
        plt.close()


def main():
    try:
        print("Starting comprehensive evaluation...")
        model_path = os.path.join("models", "ml_model.joblib")
        
        evaluator = MinimaxVsMLEvaluator(model_path=model_path)
        stats = evaluator.evaluate(num_games=500, batch_size=100)
        
        print("\nEvaluation complete. Results saved to:")
        print("- results/minimax_vs_ml_metrics.png")
        print("- Performance report shown above")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
    finally:
        if 'tracemalloc' in globals():
            tracemalloc.stop()


if __name__ == "__main__":
    main()