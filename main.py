# Connect 4 Game Implementation with AI Agents and Performance Evaluation
import os
import time
import random
import numpy as np
import sys
from connect4_game import ROWS, COLS
from connect4_agents import RandomAgent, SmartAgent, MinimaxAgent, MLAgent
from connect4_game import Connect4, PLAYER_SYMBOLS
from performance_random_vs_smart import RandomVsSmartEvaluator
from performance_smart_vs_minimax import SmartVsMinimaxEvaluator
from performance_minimax_vs_ml import MinimaxVsMLEvaluator

def clear_terminal():
    """Clears the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_agent_by_choice(choice, symbol=None):
    """Helper function to get the appropriate agent based on user choice"""
    if choice == 1:
        return RandomAgent(symbol)
    elif choice == 2:
        return SmartAgent(symbol)
    elif choice == 3:
        return MinimaxAgent(symbol, max_depth=4)
    elif choice == 4:
        return MLAgent(symbol)
    return None

def start_game(choice):
    """Main game controller function"""
    clear_terminal()
    
    # Human vs ML Agent (Option 4)
    if choice == 4:
        game = Connect4()
        human_symbol = 1
        ml_agent = MLAgent(symbol=2)

        print("=== Human vs ML Agent ===")
        print(f"You are {PLAYER_SYMBOLS[human_symbol]} (Player 1)")
        print(f"ML Agent is {PLAYER_SYMBOLS[ml_agent.symbol]} (Player 2)\n")
        game.print_board()

        while not game.is_terminal():
            if game.current_player == human_symbol:
                valid_move = False
                while not valid_move:
                    try:
                        col = input("\nYour turn! Choose column (0-6): ").strip()
                        if col.lower() in ('quit', 'exit'):
                            print("Game exited by player.")
                            return
                        col = int(col)
                        valid_move = game.make_move(col, human_symbol)
                        if not valid_move:
                            print("Invalid move! Try again.")
                    except ValueError:
                        print("Please enter a number 0-6 or 'quit'")
            else:
                # ML Agent's turn with proper console handling
                print("\nML Agent is thinking...", end='', flush=True)
                time.sleep(0.3)  # Brief pause for realistic thinking
                
                try:
                    # Get AI move (handle both dict and direct return)
                    ai_move = ml_agent.select_move(game.board, ml_agent.symbol)
                    move_col = ai_move['move'] if isinstance(ai_move, dict) else ai_move
                    
                    # Clear the "thinking" line
                    sys.stdout.write('\r' + ' ' * 50 + '\r')
                    sys.stdout.flush()
                    
                    if game.make_move(move_col, ml_agent.symbol):
                        print(f"ML Agent placed in column {move_col}")
                    else:
                        print("AI made invalid move! Trying random...")
                        valid_cols = [c for c in range(COLS) if game.board[0][c] == 0]
                        if valid_cols:
                            game.make_move(random.choice(valid_cols), ml_agent.symbol)
                except Exception as e:
                    # Clear the "thinking" line on error too
                    sys.stdout.write('\r' + ' ' * 50 + '\r')
                    sys.stdout.flush()
                    print(f"AI error: {e}. Making random move...")
                    valid_cols = [c for c in range(COLS) if game.board[0][c] == 0]
                    if valid_cols:
                        game.make_move(random.choice(valid_cols), ml_agent.symbol)

            game.print_board()

        # Game over handling
        print("\nGame Over!")
        if game.get_winner() == human_symbol:
            print(" You win! ")
        elif game.get_winner() == ml_agent.symbol:
            print("ML Agent wins!")
        else:
            print("It's a draw!")
        
        input("\nPress Enter to return to menu...")
        return

    # Human vs Minimax (Option 3) with visualization
    elif choice == 3:
        game = Connect4()
        agent = MinimaxAgent(symbol=2, max_depth=4)
        human_disc = 1
        
        print(f"=== Human vs {agent.name} ===")
        print(f"You are {PLAYER_SYMBOLS[human_disc]} (Player 1)")
        print(f"AI is {PLAYER_SYMBOLS[agent.symbol]} (Player 2)\n")
        print("Note: Minimax search tree will be visualized after each AI move")
        print("Visualizations saved to 'results/game_tree.png'\n")
        game.print_board()

        while not game.is_terminal():
            if game.current_player == human_disc:
                valid_move = False
                while not valid_move:
                    try:
                        col = int(input("\nYour turn! Choose column (0-6): "))
                        valid_move = game.make_move(col, human_disc)
                        if not valid_move:
                            print("Invalid move! Try again.")
                    except ValueError:
                        print("Please enter a number 0-6")
            else:
                print(f"\n{agent.name} is thinking...")
                # Get move with visualization
                col = agent.get_move(game.board)
                
                # Show search metrics
                print(f"  - Nodes expanded: {agent.metrics['nodes_expanded']}")
                print(f"  - Max depth reached: {agent.metrics['max_depth_reached']}")
                print(f"  - Pruning count: {agent.metrics['pruning_counts']}")
                print(f"  - Search time: {agent.metrics['execution_times'][-1]:.2f}s")
                
                game.make_move(col, agent.symbol)

            game.print_board()

        winner = game.get_winner()
        if winner == human_disc:
            print("\nYou win!")
        elif winner == agent.symbol:
            print(f"\n{agent.name} wins!")
        else:
            print("\nIt's a draw!")
        
        print("\nFinal Minimax Search Metrics:")
        print(f"Total nodes expanded: {agent.metrics['nodes_expanded']}")
        print(f"Average search depth: {np.mean(agent.metrics['iterative_deepening_depths']):.1f}")
        print(f"Total pruning count: {agent.metrics['pruning_counts']}")
        print(f"Average move time: {np.mean(agent.metrics['execution_times']):.2f}s")
        print("\nVisualizations saved to 'results/game_tree.png'")
        
        input("\nPress Enter to continue...")
        return

    # Human vs Random/Smart (Options 1-2)
    elif choice in [1, 2]:
        game = Connect4()
        agent = get_agent_by_choice(choice, symbol=2)
        human_disc = 1
        
        print(f"=== Human vs {agent.name} ===")
        print(f"You are {PLAYER_SYMBOLS[human_disc]} (Player 1)")
        print(f"AI is {PLAYER_SYMBOLS[agent.symbol]} (Player 2)\n")
        game.print_board()

        while not game.is_terminal():
            if game.current_player == human_disc:
                valid_move = False
                while not valid_move:
                    try:
                        col = int(input("\nYour turn! Choose column (0-6): "))
                        valid_move = game.make_move(col, human_disc)
                        if not valid_move:
                            print("Invalid move! Try again.")
                    except ValueError:
                        print("Please enter a number 0-6")
            else:
                print(f"\n{agent.name} is thinking...")
                col = agent.select_move(game.board, agent.symbol)
                game.make_move(col, agent.symbol)

            game.print_board()

        winner = game.get_winner()
        if winner == human_disc:
            print("\nYou win!")
        elif winner == agent.symbol:
            print(f"\n{agent.name} wins!")
        else:
            print("\nIt's a draw!")
        
        input("\nPress Enter to continue...")
        return

    # Performance Evaluations (Options 5-7)
    elif choice == 5:
        print("\nRunning Random vs Smart Evaluation (500 games)...")
        evaluator = RandomVsSmartEvaluator()
        results = evaluator.evaluate(num_games=500)
        
        print("\n=== Results ===")
        print(f"Smart Agent Decision Types: {results['smart_metrics']['decision_types']}")
        print(f"Average Decision Time: {results['smart_metrics']['avg_decision_time']:.4f}s")
        print(f"Total Evaluation Time: {results['resource_metrics']['total_time']:.2f}s")
        
        input("\nPress Enter to continue...")
        return

    elif choice == 6:
        print("\nRunning Smart vs Minimax Evaluation (100 games)...")
        evaluator = SmartVsMinimaxEvaluator(max_depth=4)
        results = evaluator.evaluate(num_games=100)
        
        print("\n=== Results ===")
        total_games = results['smart_vs_minimax']['wins'] + results['smart_vs_minimax']['losses'] + results['smart_vs_minimax']['draws']
        minimax_win_rate = (results['smart_vs_minimax']['losses'] / total_games) * 100 if total_games > 0 else 0  
        print(f"Minimax Win Rate: {minimax_win_rate:.1f}%")
        print(f"Avg Nodes Expanded: {results['minimax_metrics']['avg_nodes']:.0f}")
        print(f"Avg Search Depth: {np.mean(results['minimax_metrics']['search_depth']):.1f}")
        print(f"Pruning Rate: {results['minimax_metrics']['pruning_rate']:.1f}%")
        
        input("\nPress Enter to continue...")
        return

    elif choice == 7:
        print("\nRunning ML vs Minimax Evaluation (50 games in batches)...")
        try:
            model_path = os.path.join('models', 'ml_model.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
                
            evaluator = MinimaxVsMLEvaluator(model_path=model_path)
            stats = evaluator.evaluate(num_games=50, batch_size=50) #
            
            total_games = stats['ml_vs_minimax']['wins'] + stats['ml_vs_minimax']['losses'] + stats['ml_vs_minimax']['draws']
            minimax_win_rate = (stats['ml_vs_minimax']['losses']/total_games)*100 if total_games > 0 else 0
            draw_rate = (stats['ml_vs_minimax']['draws']/total_games)*100 if total_games > 0 else 0
            
            #visualisations are generated and saved (e.g., to 'results/minimax_vs_ml_metrics.png')
            # for inclusion in your final report or presentation. These plots typically include win distribution,
            # node expansion patterns, or inference times over time.

            print("\n=== Final Evaluation Results ===")
            print(f"Total Games: {total_games}")
            
            print("\n1. Accuracy Metrics:")
            print(f"  - ML Win Rate: {stats['ml_vs_minimax']['win_rate']:.1f}%")
            print(f"  - Minimax Win Rate: {minimax_win_rate:.1f}%")
            print(f"  - Draw Rate: {draw_rate:.1f}%")
            
            print("\n2. Game-Level Metrics:")
            print(f"  - Avg Game Length: {stats['ml_vs_minimax']['avg_game_length']:.1f} moves")
            print(f"  - Win Patterns: {dict(stats['ml_vs_minimax']['win_patterns'])}")
            
            print("\n3. Minimax Search Metrics:")
            print(f"  - Avg Nodes Expanded: {stats['minimax_metrics']['avg_nodes']:.0f}")
            print(f"  - Avg Search Depth: {np.mean(stats['minimax_metrics']['search_depth']):.1f}")
            print(f"  - Pruning Rate: {stats['minimax_metrics']['pruning_rate']:.1f}%")
            
            print("\n4. ML Agent Metrics:")
            print(f"  - Avg Inference Time: {stats['ml_metrics']['avg_time']:.4f}s")
            print(f"  - Avg Memory Usage: {stats['ml_metrics']['avg_memory']:.2f} MB")
            
            print("\n5. Resource Metrics:")
            print(f"  - Total Time: {stats['resource_metrics']['total_time']:.2f}s")
            print(f"  - Time per Game: {stats['resource_metrics']['time_per_game']:.2f}s")
            print(f"  - Peak Memory: {stats['resource_metrics']['peak_memory']:.2f} MB")
            
            print("\nVisualizations saved to 'results/minimax_vs_ml_metrics.png'")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please ensure:")
            print("1. Model is trained (run train_ml_model.py)")
            print("2. Model file exists in models/ directory")
        
        input("\nPress Enter to continue...")
        return

    elif choice == 8:
        print("Goodbye!")
        exit()

if __name__ == "__main__":
    from menu import show_menu
    while True:
        show_menu()
        try:
            choice = int(input("Enter your choice (1-8): "))
            if 1 <= choice <= 8:
                start_game(choice)
            else:
                print("Please enter a number 1-8")
        except ValueError:
            print("Invalid input. Please enter a number.")