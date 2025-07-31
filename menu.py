# menu.py
from main import start_game

def show_menu():
    print("\n" + "=" * 40)
    print("🔴🟡  WELCOME TO CONNECT 4  🟡🔴".center(40))
    print("=" * 40)
    print("Choose your AI Opponent:\n")
    print("  1.   Human vs Random Agent")
    print("  2.   Human vs Smart Agent")
    print("  3.   Human vs Minimax Agent")
    print("  4.   Human vs ML Agent")
    print("\nPerformance Evaluation:")
    print("  5.   Run Random vs Smart Evaluation")
    print("  6.   Run Smart vs Minimax Evaluation")
    print("  7.   Run ML vs Minimax Evaluation")
    print("  8.   Exit")
    print("=" * 40)

def main():
    while True:
        show_menu()
        try:
            choice = int(input("Enter your choice (1–8): "))
            if 1 <= choice <= 8:
                start_game(choice)
            else:
                print("Please enter a number from 1 to 8.")
        except ValueError:
            print("Invalid input. Enter a number from 1 to 8.")

if __name__ == "__main__":
    main()