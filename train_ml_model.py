import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

"""
This script uses the Connect-4 dataset for training a machine learning agent.

Dataset Source:
J. Tromp. "Connect-4," UCI Machine Learning Repository, 1995.
Available: https://doi.org/10.24432/C59P43
"""


# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath, header=None)
    
    # Map values
    value_map = {'x': 1, 'o': -1, 'b': 0, 'win': 1, 'loss': -1, 'draw': 0}
    for col in range(42):
        data[col] = data[col].map(value_map)
    data[42] = data[42].map(value_map)
    
    # Features and label
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    return X, y

def train_model():
    # Load and preprocess data
    X, y = load_and_preprocess_data("connect-4.data")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    
    # Save model
    model_path = os.path.join("models", "ml_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    return clf

if __name__ == "__main__":
    train_model()