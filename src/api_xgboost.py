"""FastAPI server for exoplanet classification with XGBoost model."""
import os
import io
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Classification API",
    description="XGBoost ML API for classifying exoplanet candidates using NASA Kepler data",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
scaler = None
model_path = "models/baseline.pkl"
scaler_path = "models/scaler.pkl"

# Pydantic models for request/response validation
class ExoplanetFeatures(BaseModel):
    """Features for exoplanet classification."""
    kepid: Optional[int] = None
    koi_period: Optional[float] = Field(None, description="Orbital Period [days]")
    koi_depth: Optional[float] = Field(None, description="Transit Depth [ppm]")
    koi_duration: Optional[float] = Field(None, description="Transit Duration [hours]")
    koi_impact: Optional[float] = Field(None, description="Impact Parameter")
    koi_model_snr: Optional[float] = Field(None, description="Signal-to-Noise Ratio")
    koi_steff: Optional[float] = Field(None, description="Stellar Effective Temperature [K]")
    koi_slogg: Optional[float] = Field(None, description="Stellar Surface Gravity [log10(cm/s**2)]")
    koi_srad: Optional[float] = Field(None, description="Stellar Radius [Solar radii]")
    koi_kepmag: Optional[float] = Field(None, description="Kepler-band [mag]")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Prediction: 0 (not exoplanet) or 1 (exoplanet)")
    probability: float = Field(..., description="Confidence score (0.0 to 1.0)")
    kepid: Optional[int] = Field(None, description="Kepler ID")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_samples: int

class ModelInfo(BaseModel):
    """Model information response."""
    model_path: str
    model_type: str
    is_loaded: bool
    feature_count: Optional[int] = None

def load_model():
    """Load the trained XGBoost model and scaler."""
    global model, scaler
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"XGBoost model and scaler loaded successfully")
            return True
        else:
            logger.error(f"Model or scaler files not found")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def engineer_features(features: ExoplanetFeatures) -> np.ndarray:
    """Engineer features for XGBoost model prediction."""
    # Convert to dictionary and handle None values
    feature_dict = features.model_dump()
    
    # Create feature DataFrame
    df = pd.DataFrame([feature_dict])
    
    # Basic features
    koi_period = df['koi_period'].fillna(0).infer_objects().iloc[0]
    koi_depth = df['koi_depth'].fillna(0).infer_objects().iloc[0]
    koi_duration = df['koi_duration'].fillna(0).infer_objects().iloc[0]
    koi_impact = df['koi_impact'].fillna(0.5).infer_objects().iloc[0]
    koi_model_snr = df['koi_model_snr'].fillna(10).infer_objects().iloc[0]
    koi_steff = df['koi_steff'].fillna(5800).infer_objects().iloc[0]
    koi_slogg = df['koi_slogg'].fillna(4.5).infer_objects().iloc[0]
    koi_srad = df['koi_srad'].fillna(1.0).infer_objects().iloc[0]
    koi_kepmag = df['koi_kepmag'].fillna(12).infer_objects().iloc[0]
    
    # Feature engineering (same as training)
    features_engineered = {
        'koi_period': koi_period,
        'koi_depth': koi_depth,
        'koi_duration': koi_duration,
        'koi_impact': koi_impact,
        'koi_model_snr': koi_model_snr,
        'koi_steff': koi_steff,
        'koi_slogg': koi_slogg,
        'koi_srad': koi_srad,
        'koi_kepmag': koi_kepmag,
        'transit_depth_log': np.log10(koi_depth + 1e-10),
        'period_log': np.log10(koi_period + 1e-10),
        'duration_hours': koi_duration * 24,
        'transit_speed': koi_depth / (koi_duration + 1e-10),
        'period_duration_ratio': koi_period / (koi_duration + 1e-10),
        'snr_depth_ratio': koi_model_snr / (koi_depth + 1e-10),
        'transit_quality': koi_model_snr * koi_depth
    }
    
    # Star type classification
    if koi_steff >= 4000 and koi_steff < 5000:
        star_type_encoded = 1  # K
    elif koi_steff >= 5000 and koi_steff < 6000:
        star_type_encoded = 2  # G
    elif koi_steff >= 6000 and koi_steff < 7000:
        star_type_encoded = 3  # F
    elif koi_steff >= 7000 and koi_steff < 8000:
        star_type_encoded = 4  # A
    elif koi_steff >= 8000:
        star_type_encoded = 5  # B
    else:
        star_type_encoded = 0  # M
    
    features_engineered['star_type_encoded'] = star_type_encoded
    
    # Convert to array in the same order as training
    feature_names = [
        'koi_period', 'koi_depth', 'koi_duration', 'koi_impact', 'koi_model_snr',
        'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
        'transit_depth_log', 'period_log', 'duration_hours', 'transit_speed',
        'period_duration_ratio', 'snr_depth_ratio', 'transit_quality', 'star_type_encoded'
    ]
    
    feature_array = np.array([float(features_engineered[name]) for name in feature_names]).reshape(1, -1)
    
    # Scale features
    if scaler is not None:
        feature_array = scaler.transform(feature_array)
    
    return feature_array

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Exoplanet Classification API with XGBoost",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "model_loaded": str(model is not None),
        "scaler_loaded": str(scaler is not None)
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    feature_count = None
    if model is not None:
        try:
            feature_count = model.n_features_in_
        except AttributeError:
            pass
    
    return ModelInfo(
        model_path=model_path,
        model_type="XGBoost" if model is not None else "Unknown",
        is_loaded=model is not None,
        feature_count=feature_count
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: ExoplanetFeatures):
    """Make a single prediction using XGBoost model."""
    if model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    try:
        # Engineer and preprocess features
        X = engineer_features(features)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            kepid=features.kepid
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(features_list: List[ExoplanetFeatures]):
    """Make batch predictions using XGBoost model."""
    if model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    if len(features_list) > 1000:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
    
    try:
        predictions = []
        
        for features in features_list:
            X = engineer_features(features)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0].max()
            
            predictions.append(PredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                kepid=features.kepid
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file using XGBoost model."""
    if model is None:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Limit file size
        if len(df) > 1000:
            raise HTTPException(status_code=400, detail="File too large (max 1000 rows)")
        
        # Make predictions
        predictions = []
        for _, row in df.iterrows():
            # Convert row to ExoplanetFeatures
            features_dict = row.to_dict()
            features = ExoplanetFeatures(**features_dict)
            
            X = engineer_features(features)
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0].max()
            
            predictions.append({
                "prediction": int(prediction),
                "probability": float(probability),
                "kepid": features.kepid
            })
        
        return {
            "predictions": predictions,
            "total_samples": len(predictions),
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"File prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")

@app.post("/model/reload")
async def reload_model():
    """Reload the XGBoost model."""
    success = load_model()
    if success:
        return {"message": "XGBoost model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
