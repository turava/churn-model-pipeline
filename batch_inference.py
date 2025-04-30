import pandas as pd
import joblib

input_path = "data/input_data.csv"
model_path = "model/model.pkl"
output_path = "output_predictions.csv"

print("Loading input data...")
X = pd.read_csv(input_path)
print("Loading trained model...")
model = joblib.load(model_path)
print("Generating predictions...")
preds = model.predict(X)

# Save output
pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

