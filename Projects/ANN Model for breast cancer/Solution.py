import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import requests

# Load the trained model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Load the dataset
df = pd.read_csv('data.csv')
feature_names = [
    'x.radius_mean', 'x.texture_mean', 'x.perimeter_mean', 'x.area_mean',
    'x.smoothness_mean', 'x.compactness_mean', 'x.concavity_mean',
    'x.concave_pts_mean', 'x.symmetry_mean', 'x.fractal_dim_mean',
    'x.radius_se', 'x.texture_se', 'x.perimeter_se', 'x.area_se',
    'x.smoothness_se', 'x.compactness_se', 'x.concavity_se',
    'x.concave_pts_se', 'x.symmetry_se', 'x.fractal_dim_se',
    'x.radius_worst', 'x.texture_worst', 'x.perimeter_worst', 'x.area_worst',
    'x.smoothness_worst', 'x.compactness_worst', 'x.concavity_worst',
    'x.concave_pts_worst', 'x.symmetry_worst', 'x.fractal_dim_worst'
]

# Display min and max values for each feature
print("Min and Max values for each feature (based on the dataset):\n")
min_values = df[feature_names].min()
max_values = df[feature_names].max()

for feature in feature_names:
    print(f"{feature}: Min = {min_values[feature]:.4f}, Max = {max_values[feature]:.4f}")

# Collect user input for features
user_vals = []
for feat in feature_names:
    while True:
        try:
            val = float(input(f"Enter value for {feat} (between {min_values[feat]:.4f} and {max_values[feat]:.4f}): "))
            if min_values[feat] <= val <= max_values[feat]:
                user_vals.append(val)
                break
            else:
                print(f"Value out of range! Please enter a value between {min_values[feat]:.4f} and {max_values[feat]:.4f}.")
        except ValueError:
            print("Invalid input! Please enter a valid numerical value.")

# Create a DataFrame for user input
user_df = pd.DataFrame([user_vals], columns=feature_names)

# Display min and max input values
min_val = user_df.min(axis=1).iloc[0]
max_val = user_df.max(axis=1).iloc[0]
print(f"\nMinimum input value: {min_val:.4f}")
print(f"Maximum input value: {max_val:.4f}")

# Scale the input and make predictions
scaled = scaler.transform(user_df)
pred_prob = model.predict(scaled)[0][0]
prediction = "Malignant" if pred_prob > 0.5 else "Benign"
confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

print(f"\n Predicted Cancer Type: {prediction}")
# print(f"Confidence Score: {confidence:.4f}")

# Function to get suggestions from Google Gemini API
def get_suggestions(prediction, confidence):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    api_key = "AIzaSyDqcgp6F7cXM-bzTSwOysjCgy6jDDGDuR4"  
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{
            "parts": [{
                "text": f"The model predicted the cancer type as {prediction} with a confidence score of {confidence:.4f}. Provide suggestions for the user based on this result."
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 150
        }
    }
    params = {"key": api_key}
    response = requests.post(api_url, headers=headers, json=data, params=params)
    if response.status_code == 200:
        try:
            return [response.json()['candidates'][0]['content']['parts'][0]['text']]
        except (KeyError, IndexError):
            return ["No suggestions available."]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return ["Error fetching suggestions."]

# Fetch and display suggestions
suggestions = get_suggestions(prediction, confidence)
print("\nSuggestions:")
for suggestion in suggestions:
    print(f"- {suggestion}")