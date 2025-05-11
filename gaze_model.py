import os
import pandas as pd
import numpy as np
import csv
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


class GazeRegressor:
    def __init__(self, model_type='ridge', alpha=1.0):
        """
        Initialize the GazeRegressor.
        - model_type: Choose between 'ridge', 'random_forest', or 'neural_network'.
        - alpha: Regularization parameter for Ridge regression.
        """
        self.model_type = model_type
        self.alpha = alpha
        self.model_x = None
        self.model_y = None
        self.scaler = StandardScaler()

        # Choose model type
        if model_type == 'ridge':
            self.model_x = Ridge(alpha=self.alpha)
            self.model_y = Ridge(alpha=self.alpha)
        elif model_type == 'random_forest':
            self.model_x = RandomForestRegressor(n_estimators=100)
            self.model_y = RandomForestRegressor(n_estimators=100)
        elif model_type == 'neural_network':
            from sklearn.neural_network import MLPRegressor
            self.model_x = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
            self.model_y = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose from ['ridge', 'random_forest', 'neural_network']")

    def load(self, csv_path):
        """
        Load calibration data from a CSV file and train the model.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Calibration data file not found: {csv_path}")

        try:
            data = pd.read_csv(csv_path, header=None)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")

        # Extract features and labels
        features = data.iloc[:, :12].values  # First 12 columns for landmarks
        gaze_x = data.iloc[:, 12].values  # Column 13 for x gaze
        gaze_y = data.iloc[:, 13].values  # Column 14 for y gaze

        # Normalize the features
        features = self.scaler.fit_transform(features)

        # Split into train/test sets for evaluation
        X_train, X_test, y_train_x, y_test_x, y_train_y, y_test_y = train_test_split(features, gaze_x, gaze_y, test_size=0.2, random_state=42)

        # Train models using cross-validation for better evaluation
        print(f"Training {self.model_type} model...")

        # Fit the models
        self.model_x.fit(X_train, y_train_x)
        self.model_y.fit(X_train, y_train_y)

        # Evaluate with cross-validation
        scores_x = cross_val_score(self.model_x, X_train, y_train_x, cv=5, scoring='neg_mean_squared_error')
        scores_y = cross_val_score(self.model_y, X_train, y_train_y, cv=5, scoring='neg_mean_squared_error')
        print(f"X Model MSE: {np.mean(-scores_x)}")
        print(f"Y Model MSE: {np.mean(-scores_y)}")

        # Make predictions on the test set
        pred_x = self.model_x.predict(X_test)
        pred_y = self.model_y.predict(X_test)

        # Calculate and print additional evaluation metrics
        print(f"X Model R2: {r2_score(y_test_x, pred_x)}")
        print(f"Y Model R2: {r2_score(y_test_y, pred_y)}")

        # Optionally save the models
        self.save('gaze_model_x.pkl', 'gaze_model_y.pkl', 'scaler.pkl')

    def predict(self, features):
        """
        Predicts the gaze position (x, y) based on the input features (3D landmarks).
        """
        if not isinstance(features, list) or len(features) != 12:
            raise ValueError("Features must be a list of 12 landmarks (x, y, z coordinates).")

        features = np.array(features).reshape(1, -1)  # Reshape for single prediction
        features = self.scaler.transform(features)  # Normalize the features

        predicted_x = self.model_x.predict(features)[0]
        predicted_y = self.model_y.predict(features)[0]
        return predicted_x, predicted_y

    def save(self, model_x_path, model_y_path, scaler_path):
        """
        Save the trained models and scaler to disk.
        """
        joblib.dump(self.model_x, model_x_path)
        joblib.dump(self.model_y, model_y_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Models saved to {model_x_path}, {model_y_path}, and {scaler_path}")

    def load_saved(self, model_x_path, model_y_path, scaler_path):
        """
        Load previously saved models and scaler.
        """
        if not os.path.exists(model_x_path) or not os.path.exists(model_y_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"One or more saved model files not found. Paths: {model_x_path}, {model_y_path}, {scaler_path}")

        self.model_x = joblib.load(model_x_path)
        self.model_y = joblib.load(model_y_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Loaded models from {model_x_path}, {model_y_path}, and {scaler_path}")

    def visualize(self, true_x=None, true_y=None, predicted_x=None, predicted_y=None):
        """
        Visualize the predicted vs. actual gaze points on a 2D plot.
        If true_x, true_y, predicted_x, and predicted_y are not provided, skip visualization.
        """
        if true_x is not None and true_y is not None and predicted_x is not None and predicted_y is not None:
            plt.figure(figsize=(8, 6))
            plt.scatter(true_x, true_y, color='blue', label='True Gaze')
            plt.scatter(predicted_x, predicted_y, color='red', label='Predicted Gaze')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title('True vs Predicted Gaze Positions')
            plt.legend()
            plt.show()
        else:
            print("Skipping visualization as ground truth data is not available.")


# Example Usage:
# The features (list_of_landmarks) should be extracted from the FaceTracker
# (typically from OpenSeeFace or similar)

if __name__ == "__main__":
    # Assuming calibration is already done and the CSV data is available
    gaze_regressor = GazeRegressor(model_type='ridge')  # Change model type to 'random_forest' or 'neural_network'

    # Load and train the model with your CSV calibration data
    gaze_regressor.load('calibration_data.csv')  # Replace with your actual CSV file

    # After calibration, use the FaceTracker to get 3D landmarks and predict gaze
    list_of_landmarks = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Replace with actual data
    predicted_x, predicted_y = gaze_regressor.predict(list_of_landmarks)

    # Print the predicted gaze coordinates
    print(f"Predicted Gaze: x={predicted_x}, y={predicted_y}")

    # If ground truth data is available, visualize it
    # Use actual ground truth x and y values for visualization
    true_x = [100]  # Replace with actual ground truth data
    true_y = [200]  # Replace with actual ground truth data
    predicted_x_values = [predicted_x]
    predicted_y_values = [predicted_y]
    gaze_regressor.visualize(true_x, true_y, predicted_x_values, predicted_y_values)
