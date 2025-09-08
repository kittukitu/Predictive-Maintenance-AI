🔧 Predictive Maintenance AI

An AI-powered predictive maintenance system that analyzes sensor readings (temperature, vibration, pressure, usage cycles) to predict equipment failures, optimize maintenance scheduling, and minimize costs.
It combines a Random Forest ML model with Google Gemini AI for professional recommendations.

🚀 Features

Predicts failure probability from sensor data.

Estimates failure timeline (cycles left).

Suggests maintenance schedule (immediate, preventive, or routine).

Provides cost optimization strategy.

Uses Gemini AI for professional recommendations & explanations.

Supports command-line arguments and interactive mode.

⚙️ Installation
1. Clone Repository
git clone https://github.com/yourusername/predictive-maintenance-ai.git
cd predictive-maintenance-ai

2. Create Virtual Environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt


requirements.txt

pandas
numpy
scikit-learn
google-generativeai
argparse

🔑 Setup Gemini API

Get an API Key from Google AI Studio
.

Replace your key in the script:

genai.configure(api_key="YOUR_API_KEY")

🖥️ Usage
Run with CLI arguments
python app.py --temperature 85 --vibration 0.7 --pressure 35 --usage_cycles 500

Run in interactive mode
python app.py


Example Output:

📊 Prediction Results
----------------------------------------
Failure Probability   : 0.62
Timeline              : Expected failure in ~38 cycles
Maintenance Schedule  : 🛠️ Schedule preventive maintenance soon (within 30 cycles).
Cost Optimization     : Moderate cost risk. Combine maintenance with scheduled downtime.

🤖 AI Recommendation:
Schedule preventive maintenance to avoid sudden breakdowns.

AI Explanation:
The model shows a moderate probability of failure, and preventive scheduling helps reduce costs by avoiding unplanned downtime.

✅ Test Cases
Test Case 1: Low Failure Probability
python app.py --temperature 70 --vibration 0.4 --pressure 28 --usage_cycles 200


Expected:

Failure probability low (< 0.4).

Timeline: Long (> 50 cycles).

Recommendation: Routine check only.

Test Case 2: Medium Failure Probability
python app.py --temperature 85 --vibration 0.7 --pressure 35 --usage_cycles 500


Expected:

Failure probability moderate (0.4–0.7).

Maintenance soon (~30 cycles).

AI suggests preventive maintenance.

Test Case 3: High Failure Probability
python app.py --temperature 100 --vibration 1.2 --pressure 45 --usage_cycles 800


Expected:

Failure probability high (> 0.7).

Immediate maintenance required.

AI suggests urgent part replacement.

📊 Methodology

Data: Synthetic dataset of 300 sensor readings.

Model: Random Forest Classifier (scikit-learn).

Risk Rules:

High Risk (>0.7) → Immediate maintenance.

Medium Risk (0.4–0.7) → Preventive maintenance.

Low Risk (≤0.4) → Routine checks.

AI Layer: Gemini adds professional recommendations.

📌 Roadmap

 Add real sensor integration (IoT data).

 Build Flask web dashboard.

 Add time-series failure prediction.