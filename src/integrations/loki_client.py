"""
Loki Client - Handles log collection from Grafana Loki.
"""

import os
import json
import logging
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class LokiClient:
    """Client for querying logs from Grafana Loki."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Loki client.
        
        Args:
            base_url: Base URL for Loki API (defaults to LOKI_URL from env)
            
        Raises:
            RuntimeError: If Loki is not available
        """
        self.base_url = base_url or os.getenv('LOKI_URL')
        if not self.base_url:
            raise ValueError("LOKI_URL environment variable not set")
        
        # Check if Loki is available
        if not self._check_loki_available():
            raise RuntimeError(f"Loki is not available at {self.base_url}")
        
        logger.info(f"Initialized LokiClient with base URL: {self.base_url}")
    
    def _check_loki_available(self) -> bool:
        """
        Check if Loki is available and responding.
        
        Returns:
            bool: True if Loki is available, False otherwise
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/ready")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking Loki availability: {e}")
            return False
    
    async def query_logs(self,
                        query: str,
                        start_time: datetime,
                        end_time: Optional[datetime] = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query logs from Loki.
        
        Args:
            query: LogQL query string
            start_time: Start time for the query
            end_time: End time for the query (defaults to now)
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        if end_time is None:
            end_time = datetime.now()
            
        # Convert times to nanoseconds since epoch
        start_ns = int(start_time.timestamp() * 1e9)
        end_ns = int(end_time.timestamp() * 1e9)
        
        # Build query URL
        query_url = f"{self.base_url}/loki/api/v1/query_range"
        
        # Build query parameters
        params = {
            "query": query,
            "start": start_ns,
            "end": end_ns,
            "limit": limit
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(query_url, params=params)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                logs = []
                
                # Extract log entries from response
                for result in data.get("data", {}).get("result", []):
                    for value in result.get("values", []):
                        try:
                            # Parse log entry
                            timestamp_ns = int(value[0])
                            log_entry = json.loads(value[1])
                            
                            # Convert timestamp to ISO format
                            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9)
                            
                            logs.append({
                                "timestamp": timestamp.isoformat(),
                                "level": log_entry.get("level", "INFO"),
                                "service": log_entry.get("service", "unknown"),
                                "message": log_entry.get("message", ""),
                                "labels": log_entry.get("labels", {})
                            })
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Failed to parse log entry: {e}")
                            continue
                
                logger.info(f"Retrieved {len(logs)} logs from Loki")
                return logs
                
        except httpx.HTTPError as e:
            logger.error(f"Error querying Loki: {e}")
            return []
    
    async def get_service_logs(self,
                             service: str,
                             time_window_minutes: int = 60,
                             level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logs for a specific service.
        
        Args:
            service: Service name
            time_window_minutes: Time window to look back
            level: Optional log level filter
            
        Returns:
            List of log entries
        """
        # Build LogQL query
        query = f'{{service="{service}"}}'
        if level:
            query = f'{{service="{service}",level="{level}"}}'
            
        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        return await self.query_logs(query, start_time, end_time)
    
    async def get_error_logs(self,
                           time_window_minutes: int = 60,
                           service: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get error logs.
        
        Args:
            time_window_minutes: Time window to look back
            service: Optional service filter
            
        Returns:
            List of error log entries
        """
        # Build LogQL query
        query = '{level=~"ERROR|WARN"}'
        if service:
            query = f'{{service="{service}",level=~"ERROR|WARN"}}'
            
        # Calculate time window
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        return await self.query_logs(query, start_time, end_time) 