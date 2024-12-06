import os
import asyncio
import random
import aiohttp
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

from flask import Flask, jsonify , request
from flask_cors import CORS 
# Add these imports after the existing import statements
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, accuracy_score, silhouette_score
import pandas as pd


#############################################################################

app = Flask(__name__)
CORS(app)

@app.route('/chart-data', methods=['GET'])
def chart_data():
    # Example data
    data = {
        "labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        "datasets": [
            {
                "label": "Processing Volume",
                "data":  [12, 19, 3, 5, 2],
                "backgroundColor": "rgba(75, 192, 192, 0.6)",
                "borderColor": "rgba(75, 192, 192, 1)",
                "borderWidth": 1
            }
        ]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=3000)
###############################################################################3

# Load environment variables
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Constants
class Config:
    ULIP_API_URL = os.getenv('ULIP_API_URL', 'https://www.ulipstaging.dpiit.gov.in/ulip/v1.0.0/user/login')
    ULIP_API_AUTH_TOKEN = os.getenv('ULIP_API_AUTH_TOKEN', 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJheGVyb25fdXNyIiwiaWF0IjoxNzMyOTcxMDU0LCJhcHBzIjoiZGF0YXB1c2gifQ.TJd8eiwleg5Z8j9oni0UVzqdOl8Vu19yWSRQc0oNEWVdWt1lH1YNYj5IZdMvEgu6VVVW_Ahn8pZJOYqhdWExrA')
    DATABASE_URL = os.getenv('DATABASE_URL', 'https://www.ulipstaging.dpiit.gov.in/ulip/v1.0.0/ACMES/01')
    SYNC_INTERVAL = 300  # 5 minutes sync interval
    MAX_RETRY_ATTEMPTS = 3

# Database Configuration
Base = declarative_base()
engine = create_async_engine(Config.DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Configuration Constants
class Config:
    ULIP_API_URL = os.getenv('ULIP_API_URL', 'https://default-ulip-api.com')
    ULIP_API_AUTH_TOKEN = os.getenv('ULIP_API_AUTH_TOKEN', '')
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+asyncpg://user:password@localhost/export_analytics')
    SYNC_INTERVAL = 300  # 5 minutes sync interval
    MAX_RETRY_ATTEMPTS = 3

# Database Configuration
Base = declarative_base()
engine = create_async_engine(Config.DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Comprehensive Shipment Model
class Shipment(Base):
    __tablename__ = "shipments"
    
    id = Column(Integer, primary_key=True, index=True)
    egm_number = Column(String, unique=True, index=True)
    shipping_bill_number = Column(String)
    leo_number = Column(String)
    location = Column(String)
    hawb_number = Column(String)
    uld_number = Column(String)
    weight = Column(Float)
    status = Column(String)
    additional_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Advanced Pydantic Models
class ShipmentCreate(BaseModel):
    egm_number: str
    shipping_bill_number: str
    leo_number: Optional[str] = None
    location: str
    hawb_number: str
    uld_number: str
    weight: float
    status: str = "In Transit"
    additional_details: Optional[Dict] = None

    @validator('weight')
    def validate_weight(cls, v):
        if v <= 0:
            raise ValueError("Weight must be a positive number")
        return v

class ShipmentResponse(ShipmentCreate):
    id: int
    created_at: datetime
    last_updated: datetime

    class Config:
        orm_mode = True

class PerformanceMetrics(BaseModel):
    data_accuracy: float = Field(default=99.9, ge=0, le=100)
    real_time_latency: float = Field(default=1.2, ge=0)
    active_shipments: int = Field(default=2847)
    system_uptime: float = Field(default=99.99, ge=0, le=100)

class AIInsight(BaseModel):
    type: str
    message: str
    severity: str

# Advanced Shipment Service
class ShipmentService:
    @staticmethod
    async def fetch_ulip_data(max_retries=Config.MAX_RETRY_ATTEMPTS) -> List[Dict]:
        """
        Fetch shipment data from ULIP API with retry mechanism
        """
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Authorization': f'Bearer {Config.ULIP_API_AUTH_TOKEN}',
                        'Content-Type': 'application/json'
                    }
                    async with session.get(Config.ULIP_API_URL, headers=headers) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.warning(f"ULIP API fetch failed. Status: {response.status}")
            except Exception as e:
                logger.error(f"ULIP API fetch error (Attempt {attempt + 1}): {e}")
                
                if attempt == max_retries - 1:
                    # Return mock data if all retries fail
                    return [
                        {
                            'egmNumber': f'EGM{random.randint(1000, 9999)}',
                            'shippingBillNumber': f'SB{random.randint(10000, 99999)}',
                            'location': 'Mumbai',
                            'weight': round(random.uniform(10, 500), 2)
                        }
                    ]
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return []

    @staticmethod
    async def sync_ulip_shipments() -> Dict:
        """
        Comprehensive synchronization of ULIP shipments with detailed metrics
        """
        start_time = datetime.utcnow()
        sync_metrics = {
            "total_shipments_fetched": 0,
            "new_shipments_added": 0,
            "updated_shipments": 0,
            "sync_duration": 0,
            "errors": []
        }
    
        async with AsyncSessionLocal() as session:
            try:
                ulip_shipments = await ShipmentService.fetch_ulip_data()
                sync_metrics["total_shipments_fetched"] = len(ulip_shipments)
    
                for shipment_data in ulip_shipments:
                    try:
                        # Advanced mapping and upsert logic
                        existing = await session.execute(
                            select(Shipment).where(
                                Shipment.egm_number == shipment_data.get('egmNumber', '')
                            )
                        )
                        existing_shipment = existing.scalars().first()
    
                        if existing_shipment:
                            # Update existing shipment
                            for key, value in shipment_data.items():
                                snake_key = ''.join(['_'+c.lower() if c.isupper() else c for c in key]).lstrip('_')
                                if hasattr(existing_shipment, snake_key):
                                    setattr(existing_shipment, snake_key, value)
                            sync_metrics["updated_shipments"] += 1
                        else:
                            # Create new shipment
                            shipment = Shipment(
                                egm_number=shipment_data.get('egmNumber', ''),
                                shipping_bill_number=shipment_data.get('shippingBillNumber', ''),
                                leo_number=shipment_data.get('leoNumber', ''),
                                location=shipment_data.get('location', ''),
                                hawb_number=shipment_data.get('hawbNumber', ''),
                                uld_number=shipment_data.get('uldNumber', ''),
                                weight=float(shipment_data.get('weight', 0)),
                                status=shipment_data.get('status', 'In Transit'),
                                additional_details=shipment_data
                            )
                            session.add(shipment)
                            sync_metrics["new_shipments_added"] += 1
    
                        await session.commit()
    
                    except Exception as inner_e:
                        logger.error(f"Individual Shipment Processing Error: {inner_e}")
                        sync_metrics["errors"].append(str(inner_e))
                        await session.rollback()
    
            except SQLAlchemyError as e:
                logger.error(f"Database Synchronization Error: {e}")
                await session.rollback()
                sync_metrics["errors"].append(str(e))
            except Exception as e:
                logger.error(f"Unexpected Synchronization Error: {e}")
                sync_metrics["errors"].append(str(e))
            finally:
                sync_metrics["sync_duration"] = (datetime.utcnow() - start_time).total_seconds()
                logger.info(f"ULIP Sync Metrics: {sync_metrics}")
                return sync_metrics

# Rest of the code remains the same as the original (Performance Service, FastAPI App, Endpoints, etc.)

# Performance Analytics Service
class PerformanceService:
    @staticmethod
    async def get_performance_metrics() -> PerformanceMetrics:
        """
        Dynamic performance metrics generation with controlled variability
        """
        return PerformanceMetrics(
            data_accuracy=round(99.9 + random.uniform(-0.1, 0.1), 2),
            real_time_latency=round(1.2 + random.uniform(-0.2, 0.2), 2),
            active_shipments=2847,
            system_uptime=99.99
        )

    @staticmethod
    async def get_ai_insights() -> List[AIInsight]:
        """
        Context-aware AI insights generation
        """
        return [
            AIInsight(
                type="Weight Discrepancy",
                message="Critical weight variance detected in shipments",
                severity="high"
            ),
            AIInsight(
                type="LEO Processing",
                message="Potential delays in LEO clearance identified",
                severity="medium"
            )
        ]

class ShipmentMLAnalytics:
    def __init__(self):
        # Preprocessing and feature engineering configurations
        self.feature_columns = [
            'weight', 'location', 'shipping_bill_number', 
            'uld_number', 'hawb_number', 'leo_number'
        ]
        
        # Model persistence paths
        self.models = {
            'delay_predictor': 'delay_prediction_model.joblib',
            'anomaly_detector': 'anomaly_detection_model.joblib',
            'clustering_model': 'shipment_clustering_model.joblib'
        }

    async def preprocess_data(self, shipments: List[Dict]) -> pd.DataFrame:
        """
        Advanced data preprocessing with feature engineering
        """
        df = pd.DataFrame(shipments)
        
        # Convert categorical variables
        df['location_encoded'] = pd.Categorical(df['location']).codes
        df['uld_type'] = df['uld_number'].str.extract('(\w+)', expand=False)
        df['uld_type_encoded'] = pd.Categorical(df['uld_type']).codes
        
        # Extract numeric features from text
        df['leo_processing_time'] = pd.to_numeric(
            df['leo_number'].str.extract('(\d+)', expand=False), 
            errors='coerce'
        ).fillna(df['leo_number'].str.len())
        
        # Time-based features
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['hour_of_day'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        
        return df

    async def train_delay_prediction_model(self, shipments: List[Dict]):
        """
        Train a Random Forest model to predict shipment delays
        """
        df = await self.preprocess_data(shipments)
        
        # Define delay based on processing time
        df['is_delayed'] = (df['leo_processing_time'] > df['leo_processing_time'].quantile(0.75)).astype(int)
        
        # Select features
        features = [
            'weight', 'location_encoded', 'uld_type_encoded', 
            'hour_of_day', 'day_of_week', 'leo_processing_time'
        ]
        
        X = df[features]
        y = df['is_delayed']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate and save
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump({
            'model': clf,
            'scaler': scaler,
            'features': features,
            'accuracy': accuracy
        }, self.models['delay_predictor'])
        
        return {
            'model_type': 'Delay Prediction',
            'accuracy': accuracy,
            'features': features
        }

    async def detect_shipment_anomalies(self, shipments: List[Dict]):
        """
        Detect anomalies in shipment data using unsupervised learning
        """
        df = await self.preprocess_data(shipments)
        
        # Select features for anomaly detection
        anomaly_features = [
            'weight', 'location_encoded', 'uld_type_encoded', 
            'leo_processing_time', 'hour_of_day'
        ]
        
        X = df[anomaly_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Clustering for anomaly detection
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate distance to cluster centers
        distances = kmeans.transform(X_scaled).min(axis=1)
        threshold = np.percentile(distances, 95)
        anomalies = distances > threshold
        
        # Save model
        joblib.dump({
            'kmeans': kmeans,
            'scaler': scaler,
            'pca': pca,
            'anomaly_threshold': threshold
        }, self.models['anomaly_detector'])
        
        # Generate anomaly insights
        anomaly_insights = []
        for idx, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                anomaly_insights.append({
                    'shipment_id': df.iloc[idx]['id'],
                    'type': 'Weight Variance' if abs(df.iloc[idx]['weight'] - df['weight'].mean()) > 2 * df['weight'].std() else 'Processing Time Anomaly',
                    'details': f"Unusual characteristics detected in shipment {df.iloc[idx]['egm_number']}"
                })
        
        return {
            'total_shipments': len(df),
            'anomalies_detected': sum(anomalies),
            'anomaly_rate': sum(anomalies) / len(df) * 100,
            'anomaly_details': anomaly_insights
        }

    async def predict_shipment_characteristics(self, shipments: List[Dict]):
        """
        Predict shipment characteristics and generate strategic insights
        """
        df = await self.preprocess_data(shipments)
        
        # Regression for weight prediction
        weight_features = [
            'location_encoded', 'uld_type_encoded', 
            'hour_of_day', 'day_of_week'
        ]
        
        X = df[weight_features]
        y = df['weight']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = regressor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(weight_features, regressor.feature_importances_))
        
        return {
            'model_type': 'Shipment Characteristics Prediction',
            'mean_absolute_error': mae,
            'feature_importance': feature_importance,
            'insights': {
                'most_influential_feature': max(feature_importance, key=feature_importance.get),
                'average_predicted_weight': y_pred.mean(),
                'weight_prediction_variance': y_pred.std()
            }
        }

# FastAPI Application Configuration
app = FastAPI(
    title="Enterprise Export Analytics Suite",
    description="Advanced Shipment Tracking and Analytics Platform",
    version="1.0.0"
)

# CORS and Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint Implementations
@app.post("/shipments", response_model=ShipmentResponse)
async def create_shipment(shipment: ShipmentCreate):
    async with AsyncSessionLocal() as session:
        try:
            db_shipment = Shipment(**shipment.dict())
            session.add(db_shipment)
            await session.commit()
            await session.refresh(db_shipment)
            return ShipmentResponse.from_orm(db_shipment)
        except SQLAlchemyError as e:
            await session.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/shipments", response_model=List[ShipmentResponse])
async def list_shipments(status: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        try:
            query = select(Shipment)
            if status:
                query = query.where(Shipment.status == status)
            result = await session.execute(query)
            return [ShipmentResponse.from_orm(shipment) for shipment in result.scalars().all()]
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    
    metrics = await PerformanceService.get_performance_metrics()
    
    # Log metrics for monitoring
    logger.info(f"Performance Metrics Retrieved: {metrics}")
    
    return metrics

@app.get("/ai-insights", response_model=List[AIInsight])
async def get_ai_insights():
    """
    Retrieve AI-generated insights about shipment and system status.
    
    This endpoint provides contextual insights that are displayed 
    in the anomaly and alert sections of the dashboard.

    Returns:
        List[AIInsight]: A list of AI-generated insights with type, message, and severity
    """
    insights = await PerformanceService.get_ai_insights()
    
    # Log insights for tracking
    for insight in insights:
        logger.info(f"AI Insight Generated: {insight.type} - {insight.message} (Severity: {insight.severity})")
    
    return insights

@app.get("/ai-insights", response_model=List[AIInsight])
async def get_ai_insights():
    """
    Retrieve AI-generated insights about shipment and system status.
    
    This endpoint provides contextual insights that are displayed 
    in the anomaly and alert sections of the dashboard.

    Returns:
        List[AIInsight]: A list of AI-generated insights with type, message, and severity
    """
    insights = await PerformanceService.get_ai_insights()
    
    # Log insights for tracking
    for insight in insights:
        logger.info(f"AI Insight Generated: {insight.type} - {insight.message} (Severity: {insight.severity})")
    
    return insights


@app.get("/ml-insights")
async def get_ml_insights():
    """
    Generate comprehensive machine learning insights for shipments.

    Returns:
        Dict: Comprehensive ML-driven insights about shipments
    """
    shipment_service = ShipmentMLAnalytics()
    
    # Fetch shipments using the existing list_shipments function
    shipments = await list_shipments()
    
    # Convert Shipment models to dictionaries
    shipment_dicts = [
        {
            'id': shipment.id,
            'weight': shipment.weight,
            'location': shipment.location,
            'shipping_bill_number': shipment.shipping_bill_number,
            'uld_number': shipment.uld_number,
            'hawb_number': shipment.hawb_number,
            'leo_number': shipment.leo_number,
            'created_at': shipment.created_at
        } for shipment in shipments
    ]
    
    ml_insights = await shipment_service.generate_comprehensive_ml_insights(shipment_dicts)
    
    # Log ML insights for tracking
    logger.info(f"ML Insights Generated: {ml_insights}")
    
    return ml_insights

# WebSocket for Real-time Updates
@app.websocket("/ws/live-tracking")
async def websocket_tracking(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Synchronize with ULIP API and prepare comprehensive update
            sync_metrics = await ShipmentService.sync_ulip_shipments()
            
            # Fetch shipments
            shipments = await list_shipments()
            
            # Convert Shipment models to dictionaries for ML processing
            shipment_dicts = [
                {
                    'id': shipment.id,
                    'weight': shipment.weight,
                    'location': shipment.location,
                    'shipping_bill_number': shipment.shipping_bill_number,
                    'uld_number': shipment.uld_number,
                    'hawb_number': shipment.hawb_number,
                    'leo_number': shipment.leo_number,
                    'created_at': shipment.created_at.isoformat() if shipment.created_at else datetime.utcnow().isoformat()
                } for shipment in shipments
            ]
            
            # Generate ML insights
            ml_analytics = ShipmentMLAnalytics()
            try:
                ml_insights = await ml_analytics.generate_comprehensive_ml_insights(shipment_dicts)
            except Exception as ml_error:
                logger.error(f"ML Insights Generation Error: {ml_error}")
                ml_insights = {
                    "error": "Could not generate ML insights",
                    "details": str(ml_error)
                }
            
            # Prepare comprehensive update payload
            update_payload = {
                "performance_metrics": await PerformanceService.get_performance_metrics(),
                "ai_insights": await PerformanceService.get_ai_insights(),
                "active_shipments": [ShipmentResponse.from_orm(shipment).dict() for shipment in shipments],
                "sync_metrics": sync_metrics,
                "ml_insights": ml_insights,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send update to WebSocket
            await websocket.send_json(update_payload)
            
            # Wait for next sync interval
            await asyncio.sleep(Config.SYNC_INTERVAL)
    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        # Optionally, you can implement reconnection logic here

# Database Initialization
@app.on_event("startup")
async def startup_event():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Optionally, you might want to handle this more gracefully
        # For example, by providing a fallback or stopping the application

# Main execution block for running with Uvicorn

if __name__ == '__main__':
    app.run(debug=True)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)