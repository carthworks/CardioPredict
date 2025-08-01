"""
FastAPI server for heart disease prediction
This script creates a REST API endpoint for serving ML predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="AI-powered heart disease risk assessment API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing objects
model = None
scaler = None
feature_names = None

class PatientData(BaseModel):
    """Pydantic model for patient input data validation"""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0: no, 1: yes)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0: no, 1: yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)")

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response"""
    prediction: int
    probability: float
    risk_level: str
    confidence: float

def load_model_artifacts():
    """Load the trained model and preprocessing objects"""
    global model, scaler, feature_names
    
    try:
        # Check if model files exist
        model_files = ['heart_disease_model.pkl', 'scaler.pkl', 'feature_names.pkl']
        for file in model_files:
            if not os.path.exists(file):
                logger.error(f"Model file {file} not found. Please run train_model.py first.")
                return False
        
        # Load model artifacts
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        logger.info("Model artifacts loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Features: {feature_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return False

def create_dummy_model():
    """Create a dummy model for demonstration purposes"""
    global model, scaler, feature_names
    
    logger.warning("Using dummy model for demonstration")
    
    # Create dummy objects
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    model = RandomForestClassifier(random_state=42)
    scaler = StandardScaler()
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Create dummy training data and fit
    np.random.seed(42)
    X_dummy = np.random.randn(100, len(feature_names))
    y_dummy = np.random.randint(0, 2, 100)
    
    scaler.fit(X_dummy)
    model.fit(X_dummy, y_dummy)
    
    return True

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Heart Disease Prediction API...")
    
    # Try to load real model, fallback to dummy model
    if not load_model_artifacts():
        logger.warning("Could not load trained model, using dummy model")
        create_dummy_model()
    
    logger.info("API startup complete")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": feature_names,
        "n_features": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_heart_disease(patient_data: PatientData):
    """
    Predict heart disease risk for a patient
    
    Args:
        patient_data: Patient clinical parameters
        
    Returns:
        PredictionResponse: Prediction result with risk assessment
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Convert input to DataFrame
        input_data = pd.DataFrame([patient_data.dict()])
        
        # Ensure correct feature order
        input_data = input_data[feature_names]
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Calculate risk probability and confidence
        risk_probability = prediction_proba[1]  # Probability of positive class
        confidence = max(prediction_proba)  # Confidence in prediction
        
        # Determine risk level
        if prediction == 1:
            risk_level = "High Risk"
        else:
            risk_level = "Low Risk"
        
        logger.info(f"Prediction made: {prediction}, Probability: {risk_probability:.3f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(risk_probability),
            risk_level=risk_level,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(patients_data: list[PatientData]):
    """
    Predict heart disease risk for multiple patients
    
    Args:
        patients_data: List of patient clinical parameters
        
    Returns:
        List of prediction results
    """
    try:
        if model is None or scaler is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        results = []
        
        for patient_data in patients_data:
            # Convert input to DataFrame
            input_data = pd.DataFrame([patient_data.dict()])
            input_data = input_data[feature_names]
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            risk_probability = prediction_proba[1]
            confidence = max(prediction_proba)
            risk_level = "High Risk" if prediction == 1 else "Low Risk"
            
            results.append(PredictionResponse(
                prediction=int(prediction),
                probability=float(risk_probability),
                risk_level=risk_level,
                confidence=float(confidence)
            ))
        
        logger.info(f"Batch prediction completed for {len(patients_data)} patients")
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
