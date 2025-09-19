Predictive Maintenance: Sensor-Based Failure Prediction
This project simulates daily machine sensor data, labels failure events, and applies machine learning for predictive maintenance—alerting for failures before they occur, and visualizing sensor trends and risk likelihoods.

Features
Synthetic dataset of daily vibration, temperature, and pressure sensor data for one year.

Simulated machine failures based on sensor patterns.

Target: Predict whether a machine will fail the next day (“failure_tomorrow”).

Model: Time-aware train/test split, XGBoost classification for next-day failure prediction.

User input: Predict next-day failure risk for user-entered sensor readings.

Automated maintenance recommendations based on predicted risk.

Visualizations: Sensor data over time, and the empirical distribution of “days to failure”.

Requirements
Python 3.x

pandas

numpy

xgboost

scikit-learn

matplotlib

seaborn

To install dependencies:

bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
Usage
Run the script
The script will:

Generate a yearly predictive maintenance dataset (predictive_maintenance_dataset.csv)

Train an XGBoost classifier for predicting machine failure the following day

Evaluate model accuracy and classification report on the test set

Input Sensor Values for Prediction

Enter sensor readings (vibration, temperature, pressure) at the prompt

The model will display predicted next-day failure risk probability

Recommendations

High risk (>0.7): Schedule immediate maintenance

Moderate risk (0.4–0.7): Monitor and plan maintenance

Low risk (≤0.4): Continue normal operations

Visualizations

Time-series line plot of vibration, temperature, and pressure

Distribution histogram of "days to failure" labels

Files
predictive_maintenance_dataset.csv: Generated synthetic dataset with labels, sensor readings, and days to next failure

Script file: Main executable script for data generation, modeling, risk prediction, and visualization

Notes
Train/test split preserves temporal order to avoid data leakage in time series prediction.

“Days to failure” label is simplified—real-world implementations require richer engineering and careful leakage prevention.

Visualizations require a graphical environment to display.