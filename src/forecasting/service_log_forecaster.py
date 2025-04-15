"""
Service Log Forecaster - Forecasts potential service anomalies.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

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
            log_data_path: Path to service log data (defaults to data/logs/service_logs)
        """
        self.log_data_path = Path(log_data_path) if log_data_path else Path('data/logs/service_logs')
        logger.info(f"Initializing ServiceLogForecaster with data path: {self.log_data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess service log data from JSON files.
        
        Returns:
            DataFrame containing prepared service log data
            
        Raises:
            ValueError: If no valid logs are found or if data parsing fails
        """
        try:
            # Check if directory exists
            if not self.log_data_path.exists():
                raise ValueError(f"Log data directory not found: {self.log_data_path}")
            
            all_logs = []
            # Read all JSON files in the service_logs directory
            for log_file in self.log_data_path.glob("*.json"):
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                        if isinstance(logs, list):
                            all_logs.extend(logs)
                        elif isinstance(logs, dict) and 'logs' in logs:
                            all_logs.extend(logs['logs'])
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing {log_file}: {str(e)}")
                    continue
            
            if not all_logs:
                raise ValueError("No valid logs found in service_logs directory")
            
            # Convert logs to DataFrame
            df = pd.DataFrame(all_logs)
            
            # Convert timestamp to datetime with explicit format
            if 'timestamp' in df.columns:
                try:
                    # First try parsing as Unix timestamp in milliseconds
                    df['ds'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                except (ValueError, TypeError):
                    try:
                        # Then try parsing as ISO format
                        df['ds'] = pd.to_datetime(df['timestamp'], format='ISO8601')
                    except ValueError:
                        raise ValueError(f"Unable to parse timestamps. Expected Unix milliseconds or ISO8601 format.")
            else:
                raise ValueError("No timestamp column found in log data")
            
            # Create target variable based on log levels or error counts
            if 'level' in df.columns:
                df['y'] = (df['level'].isin(['ERROR', 'WARN'])).astype(int)
            else:
                raise ValueError("No log level information found in data")
            
            # Aggregate by timestamp to get hourly counts
            df = df.groupby('ds').agg({'y': 'sum'}).reset_index()
            
            logger.info(f"Loaded {len(df)} log entries from service_logs")
            return df
            
        except Exception as e:
            logger.error(f"Error loading service log data: {str(e)}")
            raise ValueError(f"Failed to load service log data: {str(e)}")
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing when no logs are available."""
        logger.warning("Creating sample data for testing")
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7),
            end=datetime.now(),
            freq='H'
        )
        return pd.DataFrame({
            'ds': dates,
            'y': np.random.exponential(scale=2.0, size=len(dates))
        })
    
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
            
        Raises:
            ValueError: If data loading or model training fails
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Error during anomaly prediction: {str(e)}")
            raise
    
    def _save_forecast_plot(self, model: Prophet, forecast: pd.DataFrame) -> None:
        """
        Save the forecast plot to a file.
        
        Args:
            model: Trained Prophet model
            forecast: Forecast dataframe
        """
        try:
            # Create plots directory if it doesn't exist
            plots_dir = Path('data/plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plot
            fig = model.plot(forecast)
            plot_path = plots_dir / f'service_forecast_{datetime.now().strftime("%Y%m%d")}.png'
            fig.savefig(plot_path)
            plt.close(fig)
            
            logger.info(f"Forecast plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Error saving forecast plot: {str(e)}") 