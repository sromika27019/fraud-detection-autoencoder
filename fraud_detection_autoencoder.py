# Import necessary libraries
import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# First, I'll load the dataset from the path you provided
file_path = r'C:\Users\DELL\Downloads\archive (1)\creditcard.csv'  # I've updated the path to your dataset
data = pd.read_csv(file_path)

# Step 2: Feature selection and data preprocessing
# Now, I’ll separate the features (V1 to V28 and Amount) and the target (Class)
X = data.drop(columns=['Class'])
y = data['Class']

# I’ll normalize the data using StandardScaler to make sure the features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# I’ll split the dataset into training and testing sets (80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Initialize and train the AutoEncoder model
# Now, I’m going to use PyOD’s AutoEncoder model to detect anomalies (fraudulent transactions)
# I’ll set the contamination parameter to 0.01, assuming 1% of transactions are fraudulent
auto_encoder = AutoEncoder(contamination=0.01)
auto_encoder.fit(X_train)

# Step 4: Make predictions on the test data
# I’ll use the trained model to make predictions on the test set
y_pred = auto_encoder.predict(X_test)

# Step 5: Evaluate the model's performance
# To evaluate the model, I’ll print the classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Visualize the reconstruction error (for anomaly detection)
# Since AutoEncoder detects anomalies based on reconstruction error, I’ll plot the reconstruction error
reconstruction_error = auto_encoder.decision_function(X_test)

# Now, I’ll visualize the reconstruction error with a histogram to help identify patterns
plt.figure(figsize=(10,6))
plt.hist(reconstruction_error, bins=50, alpha=0.7, color='g')
plt.title("Reconstruction Error for Fraud Detection")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Optional: Saving the model and results for later use
# If I want to save the model or the scaler for future use, I can save them like this
# import joblib
# joblib.dump(auto_encoder, 'fraud_detection_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
