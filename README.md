## Anomaly Detection using LSTM, PCA, and Trivial Outlier Detection

The presented anomaly detection model leverages a combination of Long Short-Term Memory (LSTM) networks and Principal Component Analysis (PCA) to identify unusual patterns within a dataset. Anomalies, or outliers, are instances that deviate significantly from the norm and are crucial to detect for various applications such as fault detection or cybersecurity.

# Data Preprocessing:
The model begins by loading the dataset ('Dataset 1 .csv') into a Pandas DataFrame. The focus is on a subset of analog features labeled P1 through P9. To enhance model performance, the features undergo normalization using the Standard Scaler from scikit-learn, ensuring that each feature has a mean of 0 and a standard deviation of 1.

# LSTM Model Architecture:
The normalized features are then reshaped into a format suitable for univariate time series analysis using LSTM. The Sequential model from TensorFlow's Keras API is employed. The architecture consists of an LSTM layer with 50 units, using the ReLU activation function, and an output-dense layer matching the input feature dimensions. The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss. Training is conducted for 10 epochs with a batch size of 32.

# Model Training:
During training, the model learns temporal dependencies within the data, capturing sequential patterns that are crucial for detecting anomalies in time series data. The use of the MSE loss function guides the model to minimize the difference between predicted and actual values.

# Prediction and Dimensionality Reduction:
Following training, the model is employed to make predictions on the normalized features. Simultaneously, PCA is applied to the normalized features, aiming to retain 95% of the original variance. This dimensionality reduction technique is valuable for capturing essential information while reducing computational complexity.

# Combining LSTM Predictions and PCA Results:
The LSTM predictions and PCA-transformed features are concatenated to form a combined feature set. This fusion enhances the model's ability to capture both temporal and structural patterns within the data.

# Outlier Detection:
To identify anomalies, a simple Z-score-based outlier detection method is applied to the combined feature set. Z-scores quantify how many standard deviations a data point is from the mean. Instances with Z-scores exceeding a predefined threshold (0.75 in this case) in all dimensions are classified as outliers.

# Anomaly Identification:
The identified outliers are mapped back to the original dataset, allowing for the extraction and visualization of anomalous instances. These instances represent patterns that deviate significantly from the learned normal behavior.
