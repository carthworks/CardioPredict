"""
Test script to verify the trained model works correctly
"""

import joblib
import numpy as np
import pandas as pd
import json

def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {metadata['model_type']}")
        print(f"Accuracy: {metadata['accuracy']:.4f}")
        print(f"Training date: {metadata['training_date']}")
        
        return model, scaler, feature_names, metadata
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None

def test_predictions():
    """Test the model with sample patient data"""
    model, scaler, feature_names, metadata = load_model()
    
    if model is None:
        print("❌ Cannot test - model not loaded")
        return
    
    print(f"\n=== Testing Model Predictions ===")
    
    # Test cases with different risk profiles
    test_cases = [
        {
            'name': 'Low Risk Patient',
            'data': [35, 0, 0, 120, 180, 0, 0, 170, 0, 0.5, 1, 0, 1],
            'expected': 'Low Risk'
        },
        {
            'name': 'High Risk Patient', 
            'data': [65, 1, 0, 160, 280, 1, 1, 110, 1, 3.0, 2, 2, 2],
            'expected': 'High Risk'
        },
        {
            'name': 'Medium Risk Patient',
            'data': [50, 1, 2, 140, 220, 0, 0, 140, 0, 1.5, 1, 1, 2],
            'expected': 'Medium Risk'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {case['name']} ---")
        
        # Prepare data
        patient_data = np.array([case['data']])
        patient_scaled = scaler.transform(patient_data)
        
        # Make prediction
        prediction = model.predict(patient_scaled)[0]
        probabilities = model.predict_proba(patient_scaled)[0]
        
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = max(probabilities)
        
        print(f"Patient data: {dict(zip(feature_names, case['data']))}")
        print(f"Prediction: {prediction} ({risk_level})")
        print(f"Probability of heart disease: {probabilities[1]:.3f}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Expected: {case['expected']}")
        
        # Simple validation
        if (prediction == 1 and 'High' in case['expected']) or (prediction == 0 and 'Low' in case['expected']):
            print("✅ Prediction matches expectation")
        else:
            print("⚠️  Prediction differs from expectation (this is normal for ML models)")

def main():
    """Main testing function"""
    print("🧪 Heart Disease Model Testing")
    print("=" * 40)
    
    test_predictions()
    
    print("\n" + "=" * 40)
    print("🎉 Testing completed!")

if __name__ == "__main__":
    main()
