import numpy as np
import joblib

# Load saved model and scaler
# Correct path when running from project root
model = joblib.load('model/svm_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Sample input (can be replaced with dynamic input later)
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,
              0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,
              26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# Convert input to NumPy array and reshape
input_np = np.asarray(input_data).reshape(1, -1)

# Scale the input
scaled_input = scaler.transform(input_np)

# Predict
prediction = model.predict(scaled_input)

# Output result
if prediction[0] == 0:
    print("🟢 The person does NOT have Parkinson's Disease.")
else:
    print("🔴 The person HAS Parkinson's Disease.")
