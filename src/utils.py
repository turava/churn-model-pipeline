import re


def extract_last_digit(customerid):
    """Extract the last digit from a customerid if it exists."""
    return int(str(customerid)[-1]) if re.match(r'\d', str(customerid)[-1]) else None


# src/training.py
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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


def train_sgd_classifier(X_train, y_train, preprocessor):
    """Train SGDClassifier as a baseline alternative to logistic regression."""
    model = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', SGDClassifier(loss='log_loss', max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model


def optimize_hyperparameters(X_train, y_train, preprocessor):
    """Optimize logistic regression hyperparameters with GridSearchCV."""
    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'lbfgs']
    }
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def save_model(model, path):
    """Save the trained model to a given path."""
    joblib.dump(model, path)