# Komalben Suthar
# Assignment 4: Use Unsupervised Deep Learning Algorithm to Detect Fraud with PyOD

import numpy as np
import pandas as pd
import seaborn as sns
import pyod
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pyod.models.auto_encoder import AutoEncoder

# 1. Load Dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# 2. Preprocess Dataset
def preprocess_data(df):
    # Separate features and labels
    X = df.drop('Class', axis=1)  # Features
    y = df['Class']  # Labels (0 = normal, 1 = fraud)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 3. Split Dataset into train/test
def split_data(X, y):
    # Use only normal transactions for training (unsupervised)
    X_train = X[y == 0]

    # Test data contains all (normal + fraud)
    X_test, X_val, y_test, y_val = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    return X_train, X_test, y_test

# 4. Build and Train AutoEncoder Model
def train_autoencoder(X_train):
    model = AutoEncoder()
    model.fit(X_train)
    return model

# 5. Evaluate Model
def evaluate_model(model, X_test, y_test):
    # Get outlier scores (reconstruction errors)
    scores = model.decision_function(X_test)

    # Choose threshold (95th percentile of training errors)
    threshold = np.percentile(model.decision_scores_, 95)

    # Predict anomalies: 1 if score > threshold else 0
    y_pred = (scores > threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, scores):.4f}")

    return y_pred, scores, threshold

# 6. Enhanced Visualization & Analysis
def enhanced_evaluation(model, X_test, y_test, df, scores, threshold):
    plt.rcParams["figure.figsize"] = (15,8)

    # Plot anomaly scores with auto-calculated threshold
    plt.plot(scores)
    plt.axhline(y=threshold, color='r', linestyle='dotted', label='Threshold')
    plt.xlabel('Instances')
    plt.ylabel('Anomaly Scores')
    plt.title('Anomaly Scores with Threshold')
    plt.legend()
    plt.show()

    # threshold manually for experimentation
    manual_threshold = 70
    plt.plot(scores, color="green")
    plt.axhline(y=manual_threshold, color='r', linestyle='dotted', label='Manual Threshold')
    plt.xlabel('Instances')
    plt.ylabel('Anomaly Scores')
    plt.title('Anomaly Scores with Modified Threshold')
    plt.legend()
    plt.show()


    if hasattr(model, 'history_'):
        pd.DataFrame.from_dict(model.history_).plot(title='Error Loss')
        plt.show()
    else:
        print("No training history available to plot.")

    # Prepare DataFrame slice for test set for scatter plot
    if hasattr(y_test, 'index'):
        test_df = df.loc[y_test.index].copy()
    else:
        test_df = df.copy()

    test_df['anomaly_score'] = scores

    plt.figure(figsize=(15,8))
    sns.scatterplot(
        x="Time",
        y="Amount",
        hue="anomaly_score",
        palette="RdBu_r",
        size="anomaly_score",
        sizes=(20, 200),
        data=test_df
    )
    plt.xlabel('Time (seconds elapsed from first transaction)')
    plt.ylabel('Amount')
    plt.title('Scatter Plot of Transactions Colored by Anomaly Score')
    plt.legend(title='Anomaly Score', bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()

    print("\nEvaluation complete. ")

# 7. Main function
def main():
    print("Loading data...")
    data = load_data('creditcard.csv')

    print("Preprocessing data...")
    X_scaled, y = preprocess_data(data)

    print("Splitting data...")
    X_train, X_test, y_test = split_data(X_scaled, y)

    print("Training AutoEncoder model...")
    model = train_autoencoder(X_train)

    print("Evaluating model...")
    y_pred, scores, threshold = evaluate_model(model, X_test, y_test)

    print("Visualizing results...")
    enhanced_evaluation(model, X_test, y_test, data, scores, threshold)

if __name__ == "__main__":
    main()
