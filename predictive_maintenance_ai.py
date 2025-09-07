import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import google.generativeai as genai

# -----------------------
# 1. Configure Gemini API
# -----------------------
genai.configure(api_key="AIzaSyC2EVCSgC-DRWVunkKi7Ro0J1upoN3UglE")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------
# 2. Dummy Sensor Dataset
# -----------------------
np.random.seed(42)
data = pd.DataFrame({
    "temperature": np.random.normal(75, 10, 300),
    "vibration": np.random.normal(0.5, 0.2, 300),
    "pressure": np.random.normal(30, 5, 300),
    "usage_cycles": np.random.randint(100, 1000, 300),
    "failure": np.random.choice([0, 1], 300, p=[0.7, 0.3])
})

# -----------------------
# 3. Train Model
# -----------------------
X = data[["temperature", "vibration", "pressure", "usage_cycles"]]
y = data["failure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# -----------------------
# 4. Helper Functions
# -----------------------
def maintenance_schedule(prob):
    if prob > 0.7:
        return "⚠️ Immediate maintenance required (within next 10 cycles)."
    elif prob > 0.4:
        return "🛠️ Schedule preventive maintenance soon (within 30 cycles)."
    else:
        return "✅ Machine is healthy, routine check after 50+ cycles."

def cost_optimization(prob):
    if prob > 0.7:
        return "High cost risk! Replace parts now to avoid breakdown losses."
    elif prob > 0.4:
        return "Moderate cost risk. Combine maintenance with scheduled downtime."
    else:
        return "Low cost risk. Standard operational costs only."

# -----------------------
# 5. Main Function
# -----------------------
def run_prediction(temp, vib, press, cycles):
    df = pd.DataFrame([{
        "temperature": temp,
        "vibration": vib,
        "pressure": press,
        "usage_cycles": cycles
    }])

    # Predict failure probability
    failure_prob = rf_model.predict_proba(df)[0, 1]
    timeline = f"Expected failure in ~{int(100 - (failure_prob * 100))} cycles"
    schedule = maintenance_schedule(failure_prob)
    cost = cost_optimization(failure_prob)

    # AI Recommendation
    prompt = f"""
    You are an AI predictive maintenance analyst.
    Sensor readings: {df.to_dict(orient='records')[0]}
    Failure probability: {failure_prob:.2f}

    Provide:
    1. A professional maintenance recommendation.
    2. A short explanation why this recommendation is suitable.
    """
    response = model.generate_content(prompt)
    ai_text = response.text if response else "❌ No AI response"
    parts = ai_text.split("\n", 1)
    ai_recommendation = parts[0].strip() if parts else "Not generated"
    ai_explanation = parts[1].strip() if len(parts) > 1 else ai_text

    # Print results
    print("\n📊 Prediction Results")
    print("-" * 40)
    print(f"Failure Probability   : {failure_prob:.2f}")
    print(f"Timeline              : {timeline}")
    print(f"Maintenance Schedule  : {schedule}")
    print(f"Cost Optimization     : {cost}")
    print("\n🤖 AI Recommendation:")
    print(ai_recommendation)
    print("\nAI Explanation:")
    print(ai_explanation)

# -----------------------
# 6. CLI Setup
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔧 Predictive Maintenance AI (Terminal Version)")
    parser.add_argument("--temperature", type=float, help="Temperature in °C")
    parser.add_argument("--vibration", type=float, help="Vibration in m/s²")
    parser.add_argument("--pressure", type=float, help="Pressure in bar")
    parser.add_argument("--usage_cycles", type=int, help="Usage cycles count")

    args = parser.parse_args()

    # If not provided, ask interactively
    temp = args.temperature if args.temperature is not None else float(input("Enter Temperature (°C): "))
    vib = args.vibration if args.vibration is not None else float(input("Enter Vibration (m/s²): "))
    press = args.pressure if args.pressure is not None else float(input("Enter Pressure (bar): "))
    cycles = args.usage_cycles if args.usage_cycles is not None else int(input("Enter Usage Cycles: "))

    run_prediction(temp, vib, press, cycles)

