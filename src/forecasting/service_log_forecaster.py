"""
Service Log Forecaster - Forecasts potential service anomalies.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class ServiceLogForecaster:
    """Forecasts potential service anomalies based on log data."""
    
    def __init__(self, log_data_path: Optional[str] = None):
        """
        Initialize the Service Log Forecaster.
        
        Args:
            log_data_path: Path to service log data (defaults to data/service_logs.csv)
        """
        self.log_data_path = log_data_path or os.path.join('data', 'service_logs.csv')
        logger.info(f"Initializing ServiceLogForecaster with data path: {self.log_data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess service log data.
        
        Returns:
            DataFrame containing prepared service log data
        """
        try:
            # Check if file exists
            if not os.path.exists(self.log_data_path):
                logger.warning(f"Log data file not found: {self.log_data_path}")
                # Create sample data for demonstration
                return self._create_sample_data()
            
            # Load data
            df = pd.read_csv(self.log_data_path)
            logger.info(f"Loaded {len(df)} service log entries")
            
            # Preprocess data
            if 'timestamp' in df.columns:
                df['ds'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['ds'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("Data must contain 'timestamp' or 'date' column")
            
            # Ensure we have a value column
            if 'error_count' in df.columns:
                df['y'] = df['error_count']
            elif 'latency' in df.columns:
                df['y'] = df['latency']
            else:
                # If neither exists, create a synthetic metric
                df['y'] = np.random.exponential(scale=2.0, size=len(df))
                logger.warning("Created synthetic metric 'y' as no error_count or latency found")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading service log data: {str(e)}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample service log data for demonstration.
        
        Returns:
            DataFrame containing sample data
        """
        logger.info("Creating sample service log data")
        
        # Create date range for the past 60 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a dataframe with synthetic error counts
        df = pd.DataFrame({
            'ds': date_range,
            'service': 'api-service',
            'y': np.random.exponential(scale=2.0, size=len(date_range))
        })
        
        # Add some anomalies
        anomaly_indices = np.random.choice(range(len(df)), size=5, replace=False)
        for idx in anomaly_indices:
            df.loc[idx, 'y'] *= 5.0  # Make these points anomalous
        
        return df
    
    def train_model(self, data: pd.DataFrame) -> Prophet:
        """
        Train a Prophet forecasting model.
        
        Args:
            data: DataFrame containing time series data with 'ds' and 'y' columns
            
        Returns:
            Trained Prophet model
        """
        logger.info("Training Prophet model for service log forecasting")
        
        # Create and train model
        model = Prophet(
            interval_width=0.95,
            daily_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        model.fit(data[['ds', 'y']])
        logger.info("Prophet model training complete")
        
        return model
    
    def predict_anomalies(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Predict potential service anomalies for the future.
        
        Args:
            days_ahead: Number of days to forecast
            
        Returns:
            List of predicted anomalies with timestamp, severity, and description
        """
        # Load and prepare data
        data = self.load_data()
        
        # Train model
        model = self.train_model(data)
        
        # Make prediction
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        # Find anomalies (values that exceed upper bound)
        forecast['is_anomaly'] = forecast['yhat'] > forecast['yhat_upper']
        anomalies = forecast[forecast['is_anomaly']].copy()
        
        # Calculate severity based on how much the prediction exceeds the upper bound
        anomalies['severity_score'] = (anomalies['yhat'] - anomalies['yhat_upper']) / anomalies['yhat_upper']
        
        # Create result format
        result = []
        for _, row in anomalies.iterrows():
            # Determine severity
            severity = "low"
            if row['severity_score'] > 0.5:
                severity = "high"
            elif row['severity_score'] > 0.2:
                severity = "medium"
            
            # Create an anomaly entry
            anomaly = {
                "timestamp": row['ds'].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "service": "api-service",  # This would come from the actual data
                "severity": severity,
                "description": f"Potential service issue detected with confidence {row['severity_score']:.2f}",
                "forecast_value": float(row['yhat']),
                "upper_bound": float(row['yhat_upper'])
            }
            result.append(anomaly)
        
        # Save forecast plot
        self._save_forecast_plot(model, forecast)
        
        logger.info(f"Predicted {len(result)} potential service anomalies")
        return result
    
    def _save_forecast_plot(self, model: Prophet, forecast: pd.DataFrame) -> None:
        """
        Save the forecast plot to a file.
        
        Args:
            model: Trained Prophet model
            forecast: Forecast dataframe
        """
        try:
            # Create plots directory if it doesn't exist
            plots_dir = os.path.join('data', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate plot
            fig = model.plot(forecast)
            plot_path = os.path.join(plots_dir, f'service_forecast_{datetime.now().strftime("%Y%m%d")}.png')
            fig.savefig(plot_path)
            plt.close(fig)
            
            logger.info(f"Forecast plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving forecast plot: {str(e)}") 