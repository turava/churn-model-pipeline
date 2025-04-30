from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def train_logistic_regression(X_train, y_train, preprocessor):
    """Train logistic regression using preprocessing pipeline."""
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model


def save_model(model, path):
    """Save the trained model to a given path."""
    joblib.dump(model, path)
