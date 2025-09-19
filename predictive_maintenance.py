import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Generate synthetic machine sensor dataset ---
np.random.seed(42)
days = 365

dates = pd.date_range(start="2024-01-01", periods=days, freq='D')

# Simulate sensor signals (vibration, temperature, pressure)
vibration = np.random.normal(0.5, 0.1, days)
temperature = np.random.normal(75, 5, days)
pressure = np.random.normal(30, 2, days)

# Simulate failure indicator: failure probability increases with high vibration and temperature
failure_prob = (vibration - 0.4) * 3 + (temperature - 72) * 0.1 + np.random.normal(0, 0.1, days)
failure_prob = np.clip(failure_prob, 0, 1)
failures = (failure_prob > 0.6).astype(int)

# Simulate days to next failure (label) for supervised learning (simplified)
days_to_failure = np.full(days, np.nan)
last_failure_day = -1
for i in range(days):
    if failures[i] == 1:
        last_failure_day = i
    if last_failure_day == -1:
        days_to_failure[i] = np.nan
    else:
        days_to_failure[i] = max(0, last_failure_day - i)

data = pd.DataFrame({
    'date': dates,
    'vibration': vibration,
    'temperature': temperature,
    'pressure': pressure,
    'failure': failures.astype(int),
    'days_to_failure': days_to_failure
})
# For forward filling missing days_to_failure use ffill per latest pandas recommendation
data['days_to_failure'] = data['days_to_failure'].bfill().fillna(days)


# Save dataset
data.to_csv("predictive_maintenance_dataset.csv", index=False)
print("Synthetic predictive maintenance dataset saved as 'predictive_maintenance_dataset.csv'.")

# --- Step 2: Prepare data for classification: predict failure tomorrow ---
data['failure_tomorrow'] = data['failure'].shift(-1).fillna(0).astype(int)

features = ['vibration', 'temperature', 'pressure']
target = 'failure_tomorrow'

X = data[features]
y = data[target]

# Train/test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# --- Step 3: Train XGBoost classifier ---
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# --- Step 4b: User input-based failure risk prediction ---
print("\nEnter sensor readings for failure risk prediction:")

def input_float(prompt, default):
    try:
        val = input(f"{prompt} (e.g. {default:.3f}): ")
        return float(val) if val != '' else default
    except ValueError:
        print("Invalid input. Using default.")
        return default

vib_input = input_float("Vibration level", data['vibration'].iloc[-1])
temp_input = input_float("Temperature (°F)", data['temperature'].iloc[-1])
pres_input = input_float("Pressure", data['pressure'].iloc[-1])

user_sample = pd.DataFrame({
    'vibration': [vib_input],
    'temperature': [temp_input],
    'pressure': [pres_input]
})

user_pred_prob = model.predict_proba(user_sample)[0][1]
print(f"\nPredicted Failure Risk based on input: {user_pred_prob:.3f}")

print("\nMaintenance Recommendation:")
if user_pred_prob > 0.7:
    print("- High failure risk: Schedule immediate maintenance and inspection.")
elif user_pred_prob > 0.4:
    print("- Moderate risk: Monitor closely and plan maintenance soon.")
else:
    print("- Low risk: Continue normal operation.")

# --- Step 5: Visualization ---
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['vibration'], label='Vibration')
plt.plot(data['date'], data['temperature'], label='Temperature')
plt.plot(data['date'], data['pressure'], label='Pressure')
plt.axvline(x=data['date'].iloc[-1], color='r', linestyle='--', label='Today')
plt.title('Sensor Data over Time')
plt.xlabel('Date')
plt.ylabel('Sensor Readings')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data, x='days_to_failure', bins=30, kde=True)
plt.title('Distribution of Days to Failure')
plt.xlabel('Days to Failure')
plt.tight_layout()
plt.show()
