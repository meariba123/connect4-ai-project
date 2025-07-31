# how to run the code
download the file
unzip the file
and you get all te files needed 
 
folders:
3011163_Ariba (inside that folder there is main folder and the models plus results folder inside that main folder)
main
models
results

files:
connect4_agents.py
connect4_game.py
main.py
menu.py -> the run up file it starts from!
performance_minimax_vs_ml.py
performance_random_vs_smart.py
performance_smart_vs_minimax.py
performance_utils.py
train_ml_model.py
smart_vs_minimax_results.csv

# how to play the game
you have a set of options 1-8
1-4 are the humans v agents
1 - human v random agent
2 - human v smart agent
3 - human v minimax agent
4 - human v ml agent

yellow disc is human 
red disc is AI agent
its a 6x7 board

Technologies Used
Core Programming
Python 3.8+ (Main language)

NumPy (For board calculations)

Matplotlib (For performance graphs)

AI & Machine Learning
scikit-learn (Random Forest ML model)

joblib (To save/load trained models)

Minimax Algorithm (With alpha-beta pruning)


then 5-7 are the performance evaluations of each of the agents 
5 -  Run Random vs Smart Evaluation
6 -  Run Smart vs Minimax Evaluation
7 -  Run ML vs Minimax Evaluation

and the 8th option is to exit the game

running the evaluations may take some time due to all the evaluations points being processed and the 7th option may take take 10-15 minutes as it batches 100 games in 5 batches. 
in the 12 minute video i used 50 games as an example to show the logic and implementation behind it but it works for 500 games and you can test that out when running the game.

the graphs and charts made from the performance evaluation etc are saved into the results folder 
and the trained model is stored in the models folder 

references: 

J. Tromp. "Connect-4," UCI Machine Learning Repository, 1995. [Online]. Available:
https://doi.org/10.24432/C59P43.

For tracking memory usage (https://docs.python.org/3/library/tracemalloc.html)

 Machine Learning agent using a pre-trained model for Connect Four.
    References:
    - Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"
    - Géron, A. (2019). "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"


https://stackoverflow.com/questions/67730079/how-to-write-a-enumerate-loop

pruning rates: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

Get current memory usage in MB https://stackoverflow.com/questions/70525623/measuring-the-allocated-memory-with-tracemalloc

Game Length Histogram (https://www.w3schools.com/python/matplotlib_histograms.asp)

anymore references will be written within the code