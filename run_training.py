import pandas as pd
from src.preprocessing import load_and_merge_data, filter_by_customerid, select_features, build_preprocessing_pipeline
from src.training import train_logistic_regression, save_model
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Load and merge all datasets
print("Loading and merging data...")
df = load_and_merge_data()

# Filter by last digit of customerid (replace with your actual digits)
valid_digits = ['1', '3', '5']  # Replace these with digits from your student ID
df = filter_by_customerid(df, valid_digits)

# Drop rows where target is missing (required for stratify)
df = df.dropna(subset=['target'])

# Separate target
y = df['target']
#X = select_features(df.drop(columns=['target', 'customerid']))
X = select_features(df)

# Define types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Preprocessing pipeline
preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
print("Training logistic regression model...")
model = train_logistic_regression(X_train, y_train, preprocessor)

# Evaluate
print("Evaluating model...")
metrics = evaluate_model(model, X_test, y_test)
print(metrics)

# Save model
print("Saving model...")
save_model(model, "model/model.pkl")
print("All done.")
