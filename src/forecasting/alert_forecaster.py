"""
Alert Forecaster - Forecasts potential system alerts.
"""

import os
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class AlertForecaster:
    """Forecasts potential system alerts based on historical alert data."""
    
    def __init__(self, alert_data_path: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize the Alert Forecaster.
        
        Args:
            alert_data_path: Path to historical alert data (defaults to data/alerts.csv)
            model_path: Path for saving/loading ML model (defaults to data/models/alert_model.pkl)
        """
        self.alert_data_path = alert_data_path or os.path.join('data', 'alerts.csv')
        self.model_path = model_path or os.path.join('data', 'models', 'alert_model.pkl')
        logger.info(f"Initializing AlertForecaster with data path: {self.alert_data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess alert data.
        
        Returns:
            DataFrame containing prepared alert data
        """
        try:
            # Check if file exists
            if not os.path.exists(self.alert_data_path):
                logger.warning(f"Alert data file not found: {self.alert_data_path}")
                # Create sample data for demonstration
                return self._create_sample_data()
            
            # Load data
            df = pd.read_csv(self.alert_data_path)
            logger.info(f"Loaded {len(df)} alert entries")
            
            # Preprocess data
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("Data must contain 'timestamp' or 'date' column")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading alert data: {str(e)}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample alert data for demonstration.
        
        Returns:
            DataFrame containing sample data
        """
        logger.info("Creating sample alert data")
        
        # Create date range for the past 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random features
        n_samples = len(date_range)
        df = pd.DataFrame({
            'timestamp': date_range,
            'cpu_usage': np.random.uniform(10, 90, n_samples),
            'memory_usage': np.random.uniform(20, 95, n_samples),
            'request_count': np.random.poisson(1000, n_samples),
            'error_rate': np.random.beta(2, 10, n_samples),
            'response_time': np.random.gamma(2, 0.2, n_samples)
        })
        
        # Generate some alert incidents
        df['incident'] = 0
        
        # Create incidents based on rules
        for i in range(n_samples):
            if (df.loc[i, 'cpu_usage'] > 85 and df.loc[i, 'memory_usage'] > 90) or \
               (df.loc[i, 'error_rate'] > 0.3) or \
               (df.loc[i, 'response_time'] > 1.0 and df.loc[i, 'request_count'] > 1200):
                df.loc[i, 'incident'] = 1
        
        return df
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variable for model training.
        
        Args:
            data: DataFrame containing alert data
            
        Returns:
            Tuple of (features array, target array)
        """
        # Select features
        feature_cols = ['cpu_usage', 'memory_usage', 'request_count', 'error_rate', 'response_time']
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in data.columns:
                data[col] = np.random.random(len(data))
                logger.warning(f"Created synthetic feature '{col}'")
        
        # Create target
        if 'incident' not in data.columns:
            # Generate synthetic target if missing
            high_load = (data['cpu_usage'] > 80) & (data['memory_usage'] > 85)
            high_errors = data['error_rate'] > 0.25
            data['incident'] = ((high_load | high_errors)).astype(int)
            logger.warning("Created synthetic target 'incident'")
        
        X = data[feature_cols].values
        y = data['incident'].values
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """
        Train an alert prediction model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training alert prediction model")
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train model
        model.fit(X, y)
        logger.info("Alert prediction model training complete")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {self.model_path}")
        
        return model
    
    def evaluate_model(self, model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Args:
            model: Trained model
            X: Test features
            y: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model accuracy: {report['accuracy']:.2f}")
        return report
    
    def predict_alerts(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Predict potential system alerts for the future.
        
        Args:
            days_ahead: Number of days to forecast
            
        Returns:
            List of predicted alerts with probability and recommendations
        """
        # Load data
        data = self.load_data()
        
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Train or load model
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model, training new one: {str(e)}")
                model = self.train_model(X, y)
        else:
            model = self.train_model(X, y)
        
        # Evaluate model
        eval_metrics = self.evaluate_model(model, X, y)
        
        # Generate future data points (could be more sophisticated)
        future_data = []
        current_date = datetime.now()
        
        for i in range(days_ahead):
            future_date = current_date + timedelta(days=i+1)
            
            # Get the last 7 days of data to base predictions on
            recent_data = data.iloc[-7:].copy() if len(data) >= 7 else data.copy()
            
            # Create a new data point with some random variation from recent trends
            new_point = {
                'timestamp': future_date,
                'cpu_usage': recent_data['cpu_usage'].mean() + np.random.normal(0, 5),
                'memory_usage': recent_data['memory_usage'].mean() + np.random.normal(0, 5),
                'request_count': recent_data['request_count'].mean() + np.random.normal(0, 100),
                'error_rate': max(0, recent_data['error_rate'].mean() + np.random.normal(0, 0.05)),
                'response_time': max(0.1, recent_data['response_time'].mean() + np.random.normal(0, 0.1))
            }
            
            future_data.append(new_point)
        
        # Convert to DataFrame
        future_df = pd.DataFrame(future_data)
        
        # Prepare features for prediction
        future_X, _ = self.prepare_features(future_df)
        
        # Make predictions
        probabilities = model.predict_proba(future_X)[:, 1]  # Probability of class 1 (incident)
        
        # Generate results
        results = []
        for i, prob in enumerate(probabilities):
            if prob > 0.3:  # Only include predictions with some probability
                start_time = future_data[i]['timestamp']
                end_time = start_time + timedelta(hours=12)  # Assume alert lasts for 12 hours
                
                # Identify services affected based on mock rules
                affected_services = []
                
                if future_data[i]['cpu_usage'] > 75:
                    affected_services.append('computing-service')
                if future_data[i]['memory_usage'] > 80:
                    affected_services.append('api-gateway')
                if future_data[i]['error_rate'] > 0.2:
                    affected_services.append('authentication-service')
                if future_data[i]['response_time'] > 0.8:
                    affected_services.append('database-service')
                
                if not affected_services:
                    affected_services = ['unknown-service']
                
                # Generate recommendations based on affected metrics
                recommendations = []
                if future_data[i]['cpu_usage'] > 75:
                    recommendations.append("Monitor CPU usage and consider scaling compute resources")
                if future_data[i]['memory_usage'] > 80:
                    recommendations.append("Check for memory leaks and consider increasing memory allocation")
                if future_data[i]['error_rate'] > 0.2:
                    recommendations.append("Investigate error logs for recurring patterns")
                if future_data[i]['response_time'] > 0.8:
                    recommendations.append("Review database query performance")
                
                recommendation = "; ".join(recommendations) if recommendations else "Monitor system performance closely"
                
                result = {
                    "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "incident_probability": float(prob),
                    "affected_services": affected_services,
                    "recommendation": recommendation,
                    "metrics": {
                        "cpu_usage": float(future_data[i]['cpu_usage']),
                        "memory_usage": float(future_data[i]['memory_usage']),
                        "error_rate": float(future_data[i]['error_rate']),
                        "response_time": float(future_data[i]['response_time'])
                    }
                }
                results.append(result)
        
        # Save prediction plot
        self._save_prediction_plot(future_df, probabilities)
        
        logger.info(f"Predicted {len(results)} potential alerts")
        return results
    
    def _save_prediction_plot(self, future_df: pd.DataFrame, probabilities: np.ndarray) -> None:
        """
        Save the alert prediction plot to a file.
        
        Args:
            future_df: DataFrame with future data points
            probabilities: Array of predicted probabilities
        """
        try:
            # Create plots directory if it doesn't exist
            plots_dir = os.path.join('data', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(future_df['timestamp'], probabilities, marker='o')
            plt.axhline(y=0.5, color='r', linestyle='--', label='Alert Threshold')
            plt.xlabel('Date')
            plt.ylabel('Alert Probability')
            plt.title('Alert Probability Forecast')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            plot_path = os.path.join(plots_dir, f'alert_forecast_{datetime.now().strftime("%Y%m%d")}.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Alert prediction plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving prediction plot: {str(e)}") 