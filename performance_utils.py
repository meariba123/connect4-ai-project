#importing necessary libraries
import matplotlib.pyplot as plt  # used to create plots and charts
import numpy as np  # used for numerical operations (e.g., averaging game length)
from collections import defaultdict  # allows dictionary with default values
import csv  # used for saving performance results to CSV files
import time  # used for tracking time taken
import tracemalloc  # used to trace memory allocation (helps measure peak memory)

#creating a utility class for performance-related functions and common game mechanics
class PerformanceUtils:
    ROWS, COLS = 6, 7  # standard Connect 4 board size

    @staticmethod
    def create_new_board():
        # returns a new 6x7 board filled with zeros (0 = empty cell)
        return [[0] * PerformanceUtils.COLS for _ in range(PerformanceUtils.ROWS)]

    @staticmethod
    def make_move(board, col, disc):
        #places the disc in the lowest available row in the given column
        for row in range(PerformanceUtils.ROWS - 1, -1, -1):  # start from bottom row
            if board[row][col] == 0:
                board[row][col] = disc
                return True  # move successful
        return False  # column is full, move not possible

    @staticmethod
    def check_for_win(board, disc):
        # checks if the given disc has won the game (4 in a row)

        #checks horizontal wins
        for row in range(PerformanceUtils.ROWS):
            for col in range(PerformanceUtils.COLS - 3):
                if all(board[row][col + i] == disc for i in range(4)):
                    return True

        # Check vertical wins
        for row in range(PerformanceUtils.ROWS - 3):
            for col in range(PerformanceUtils.COLS):
                if all(board[row + i][col] == disc for i in range(4)):
                    return True

        # Check diagonal wins (top-left to bottom-right)
        for row in range(PerformanceUtils.ROWS - 3):
            for col in range(PerformanceUtils.COLS - 3):
                if all(board[row + i][col + i] == disc for i in range(4)):
                    return True

        # Check diagonal wins (bottom-left to top-right)
        for row in range(3, PerformanceUtils.ROWS):
            for col in range(PerformanceUtils.COLS - 3):
                if all(board[row - i][col + i] == disc for i in range(4)):
                    return True

        return False  # no win found

    @staticmethod
    def is_draw(board):
        # checks if the board is full (draw = no more valid moves)
        return all(board[0][col] != 0 for col in range(PerformanceUtils.COLS))

    @staticmethod
    def init_matchup_stats():
        # initializes the data structure for storing performance statistics
        return {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'game_lengths': [],
            'win_patterns': defaultdict(int)  # you can track patterns of wins
        }

    @staticmethod
    def visualize_matchup(stats, matchup_key, title):
        """creates visualizations for matchup results"""
        plt.figure(figsize=(12, 5))  # setting figure size for better layout

        #win/Loss/Draw Pie Chart
        plt.subplot(1, 2, 1)
        plt.pie(
            [stats[matchup_key]['wins'], stats[matchup_key]['losses'], stats[matchup_key]['draws']],
            labels=['Wins', 'Losses', 'Draws'],
            autopct='%1.1f%%',
            colors=['green', 'red', 'blue']
        )
        plt.title(f'{title} Results')

        #win Patterns Bar Chart
        plt.subplot(1, 2, 2)
        patterns, counts = zip(*stats[matchup_key]['win_patterns'].items())
        plt.bar(patterns, counts, color='orange')
        plt.title('Winning Patterns')

        plt.tight_layout()  # adjust layout so nothing overlaps
        plt.show()  # display the plot

        # Game Length Histogram (https://www.w3schools.com/python/matplotlib_histograms.asp)
        plt.figure(figsize=(8, 4))
        plt.hist(stats[matchup_key]['game_lengths'], bins=20, color='purple', alpha=0.7)
        plt.title('Game Length Distribution')
        plt.xlabel('Moves')
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def save_results(filename, stats, matchup_key):
        """saves the game results to a CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Game', 'Winner', 'Moves', 'WinPattern'])  # header

            stats = stats[matchup_key]
            for i in range(len(stats['game_lengths'])):
                # determining the result type based on game index
                result = 'Win' if i < stats['wins'] else 'Loss' if i < stats['wins'] + stats['losses'] else 'Draw'
                
                # choosing a win pattern
                if result != 'Draw' and stats['win_patterns']:
                    pattern = list(stats['win_patterns'].keys())[i % len(stats['win_patterns'])]
                else:
                    pattern = 'N/A'
                
                # writing the row to CSV
                writer.writerow([i + 1, result, stats['game_lengths'][i], pattern])
