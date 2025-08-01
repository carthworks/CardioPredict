"""
Heart Disease Prediction Model Training Script
This script trains a machine learning model to predict heart disease risk.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """
    Load and prepare the heart disease dataset.
    For this example, we'll create a synthetic dataset similar to the UCI Heart Disease dataset.
    In production, you would load the actual dataset.
    """
    print("Loading heart disease dataset...")
    
    # Create synthetic data similar to UCI Heart Disease dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(25, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),  # 0: female, 1: male
        'cp': np.random.randint(0, 4, n_samples),   # chest pain type
        'trestbps': np.random.randint(90, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(120, 400, n_samples),     # cholesterol
        'fbs': np.random.randint(0, 2, n_samples),  # fasting blood sugar > 120
        'restecg': np.random.randint(0, 3, n_samples),      # resting ECG
        'thalach': np.random.randint(60, 220, n_samples),   # max heart rate
        'exang': np.random.randint(0, 2, n_samples),        # exercise induced angina
        'oldpeak': np.random.uniform(0, 6, n_samples),      # ST depression
        'slope': np.random.randint(0, 3, n_samples),        # slope of peak exercise ST
        'ca': np.random.randint(0, 4, n_samples),           # number of major vessels
        'thal': np.random.randint(1, 4, n_samples)          # thalassemia
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on risk factors (synthetic logic)
    risk_score = (
        (df['age'] > 55) * 0.3 +
        (df['sex'] == 1) * 0.2 +
        (df['cp'] > 2) * 0.25 +
        (df['trestbps'] > 140) * 0.2 +
        (df['chol'] > 240) * 0.15 +
        (df['fbs'] == 1) * 0.1 +
        (df['thalach'] < 120) * 0.2 +
        (df['exang'] == 1) * 0.15 +
        (df['oldpeak'] > 2) * 0.2 +
        (df['ca'] > 0) * 0.25 +
        np.random.normal(0, 0.1, n_samples)  # add some noise
    )
    
    df['target'] = (risk_score > 0.5).astype(int)
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n=== Data Exploration ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def preprocess_data(df):
    """Preprocess the data for training"""
    print("\n=== Data Preprocessing ===")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_models(X_train, y_train):
    """Train multiple models and select the best one"""
    print("\n=== Model Training ===")
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        
        print(f"{name} - CV Accuracy: {mean_cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with CV accuracy: {best_score:.4f}")
    
    # Train the best model on full training set
    best_model.fit(X_train, y_train)
    
    return best_model, best_name

def hyperparameter_tuning(model, X_train, y_train, model_name):
    """Perform hyperparameter tuning for the best model"""
    print(f"\n=== Hyperparameter Tuning for {model_name} ===")
    
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:  # Logistic Regression
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the trained model"""
    print("\n=== Model Evaluation ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return accuracy

def save_model(model, scaler, feature_names):
    """Save the trained model and preprocessing objects"""
    print("\n=== Saving Model ===")
    
    # Save model
    joblib.dump(model, 'heart_disease_model.pkl')
    print("Model saved as 'heart_disease_model.pkl'")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
    
    # Save feature names
    joblib.dump(list(feature_names), 'feature_names.pkl')
    print("Feature names saved as 'feature_names.pkl'")
    
    # Save model metadata
    metadata = {
        'model_type': type(model).__name__,
        'features': list(feature_names),
        'n_features': len(feature_names),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata saved as 'model_metadata.json'")

def main():
    """Main training pipeline"""
    print("=== Heart Disease Prediction Model Training ===")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Explore data
    df = explore_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train models
    best_model, best_name = train_models(X_train, y_train)
    
    # Hyperparameter tuning
    tuned_model = hyperparameter_tuning(best_model, X_train, y_train, best_name)
    
    # Evaluate model
    accuracy = evaluate_model(tuned_model, X_test, y_test, feature_names)
    
    # Save model
    save_model(tuned_model, scaler, feature_names)
    
    print(f"\n=== Training Complete ===")
    print(f"Final model accuracy: {accuracy:.4f}")
    print("Model files saved successfully!")

if __name__ == "__main__":
    main()
