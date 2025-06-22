# Fraud Detection Using AutoEncoder with PyOD
Overview
This project implements a fraud detection system using an unsupervised deep learning approach with an AutoEncoder model provided by the PyOD library. The goal is to identify fraudulent credit card transactions by learning normal transaction patterns and detecting anomalies as deviations.

The model leverages reconstruction errors to flag unusual transactions, enabling the detection of novel fraud cases without relying on labeled data.

Dataset
The project uses the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

The dataset contains anonymized features (V1, V2, ..., V28), a Time feature, Amount, and a Class label (0 = normal, 1 = fraud).

This will:

Load and preprocess the data

Train the AutoEncoder on normal transactions

Evaluate the model on the test set

Print classification metrics and AUC-ROC score

Show a histogram plot of reconstruction errors for normal and fraudulent transactions

Results
The model achieves an AUC-ROC score around 0.95, demonstrating strong ability to distinguish fraud from normal transactions.

The classification report shows precision, recall, and F1-score, which are crucial metrics for fraud detection.

Visualization helps interpret the separation between normal and fraud in reconstruction errors.

References
PyOD Library Documentation
Credit Card Fraud Detection Dataset on Kaggle
Scikit-learn Documentation
Matplotlib Documentation

