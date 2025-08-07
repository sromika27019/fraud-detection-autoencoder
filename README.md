# fraud-detection-autoencoder
# Fraud Detection Using AutoEncoder

*This repository contains a machine learning model for fraud detection using AutoEncoder, implemented with the PyOD library. I use this model to identify fraudulent transactions from an anonymized credit card transactions dataset.*

## Overview

*In this project, I build and train an anomaly detection model using the AutoEncoder technique from PyOD. AutoEncoder is particularly useful for fraud detection, as it identifies outliers by calculating the reconstruction error. Transactions with high reconstruction errors are more likely to be fraudulent. The dataset used in this project is from Kaggle, containing anonymized credit card transaction data.*

### Objective:
*The objective of this project is to detect fraudulent transactions by training an anomaly detection model and evaluating its performance on the test set.*

---

## Files in this Repository

- `fraud_detection_autoencoder.py`: The Python script containing the code for loading, processing the data, training the AutoEncoder model, and evaluating its performance.
- `creditcard.csv`: The dataset of anonymized credit card transactions (ensure the file path is correct when running the script on your local machine).
- `README.md`: This file containing the project description and setup instructions.

---

## Requirements

Before running the code, ensure that you have the following libraries installed:

- `pandas`
- `numpy`
- `pyod`
- `scikit-learn`
- `matplotlib`
- `seaborn` (for visualizing the confusion matrix)

You can install these libraries using pip:

