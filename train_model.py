"""
Disease Prediction System - Model Training Script
====================================================
This script handles data preprocessing, model training, evaluation,
and serialization for the Disease Prediction System.

Author: Research Project
Date: 2026
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Configuration
DATASET_PATH = 'dataset.csv'
MODEL_PATH = 'disease_prediction_model.pkl'
TEST_SIZE = 0.2  # 80% training, 20% testing
RANDOM_STATE = 42


def load_dataset(filepath):
    """
    Load the symptom-disease dataset from CSV file.
    
    Args:
        filepath: Path to the dataset CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns) - 1}")
    print(f"Disease classes: {df['Disease'].nunique()}")
    return df


def preprocess_data(df):
    """
    Preprocess the dataset for model training.
    
    Steps:
    1. Check for missing values
    2. Separate features and target
    3. Convert to appropriate data types
    
    Args:
        df: Raw dataset DataFrame
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of symptom names
        target_names: List of disease names
    """
    print("\n--- Data Preprocessing ---")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values found: {missing_values}")
    
    if missing_values > 0:
        print("Removing rows with missing values...")
        df = df.dropna()
        print(f"Dataset shape after removal: {df.shape}")
    
    # Separate features and target
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    feature_names = X.columns.tolist()
    target_names = y.unique().tolist()
    
    print(f"Features (symptoms): {len(feature_names)}")
    print(f"Target classes (diseases): {len(target_names)}")
    
    return X, y, feature_names, target_names


def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    print("\n--- Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y  # Ensure balanced class distribution
    )
    
    print(f"Training set size: {len(X_train)} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"Testing set size: {len(X_test)} ({TEST_SIZE*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Decision Tree classifier on the training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained Decision Tree classifier
    """
    print("\n--- Model Training ---")
    
    # Initialize Decision Tree classifier
    # Using entropy as splitting criterion for better information gain
    clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=15,  # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    
    print("Training Decision Tree classifier...")
    clf.fit(X_train, y_train)
    print("Model training completed!")
    
    return clf


def evaluate_model(clf, X_test, y_test, target_names):
    """
    Evaluate the trained model on the test set.
    
    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels
        target_names: List of disease names
    """
    print("\n--- Model Evaluation ---")
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix shape: {cm.shape}")
    
    return accuracy


def save_model(clf, feature_names, target_names, filepath):
    """
    Serialize the trained model using pickle.
    
    Args:
        clf: Trained classifier
        feature_names: List of symptom names
        target_names: List of disease names
        filepath: Path to save the model
    """
    print("\n--- Model Serialization ---")
    
    model_data = {
        'model': clf,
        'feature_names': feature_names,
        'target_names': target_names
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {filepath}")


def visualize_tree(clf, feature_names, target_names):
    """
    Visualize the Decision Tree (optional).
    
    Args:
        clf: Trained classifier
        feature_names: List of symptom names
        target_names: List of disease names
    """
    print("\n--- Tree Visualization ---")
    
    plt.figure(figsize=(30, 20))
    tree.plot_tree(
        clf,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3  # Only show top 3 levels for readability
    )
    plt.title('Decision Tree Visualization (Top 3 Levels)', fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
    print("Tree visualization saved to: decision_tree_visualization.png")
    plt.close()


def get_feature_importance(clf, feature_names):
    """
    Get and display feature importance scores.
    
    Args:
        clf: Trained classifier
        feature_names: List of symptom names
    """
    print("\n--- Feature Importance (Top 10) ---")
    
    importance = pd.DataFrame({
        'Symptom': feature_names,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(importance.head(10).to_string(index=False))
    
    return importance


def main():
    """
    Main function to execute the complete training pipeline.
    """
    print("=" * 60)
    print("DISEASE PREDICTION SYSTEM - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load dataset
    df = load_dataset(DATASET_PATH)
    
    # Step 2: Preprocess data
    X, y, feature_names, target_names = preprocess_data(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Train model
    clf = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    accuracy = evaluate_model(clf, X_test, y_test, target_names)
    
    # Step 6: Get feature importance
    importance = get_feature_importance(clf, feature_names)
    
    # Step 7: Save model
    save_model(clf, feature_names, target_names, MODEL_PATH)
    
    # Step 8: Visualize tree (optional)
    try:
        visualize_tree(clf, feature_names, target_names)
    except Exception as e:
        print(f"Tree visualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print("=" * 60)
    
    return clf, feature_names, target_names


if __name__ == "__main__":
    main()
