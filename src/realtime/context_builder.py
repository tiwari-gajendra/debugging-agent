"""
Context Builder - Collects and organizes contextual information for debugging.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document

# Import integrations
from src.integrations.loki_client import LokiClient

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Builds comprehensive context for debugging by collecting and organizing
    relevant information from various sources.
    """
    
    def __init__(self, 
                log_source: str = "loki",
                metrics_source: str = "prometheus",
                traces_source: str = "jaeger",
                vector_db_path: Optional[str] = None):
        """
        Initialize the ContextBuilder agent.
        
        Args:
            log_source: Source of logs (loki, cloudwatch, etc.)
            metrics_source: Source of metrics (prometheus, cloudwatch, etc.)
            traces_source: Source of traces (jaeger, x-ray, etc.)
            vector_db_path: Path to the vector database for RAG
            
        Raises:
            RuntimeError: If no log source is available
        """
        self.log_source = log_source
        self.metrics_source = metrics_source
        self.traces_source = traces_source
        self.vector_db_path = vector_db_path or os.path.join('data', 'vector_store')
        
        # Initialize Loki client if using Loki
        self.loki_client = None
        if log_source == "loki":
            try:
                self.loki_client = LokiClient()
                logger.info(f"Successfully connected to Loki at {self.loki_client.base_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize Loki client: {e}")
                logger.info("Falling back to service_logs directory")
                self.log_source = "service_logs"
        
        # Check service_logs directory if using service_logs
        if self.log_source == "service_logs":
            service_logs_dir = Path("data/logs/service_logs")
            if not service_logs_dir.exists():
                raise RuntimeError("No log source available: Loki is not available and service_logs directory does not exist")
            
            # Check if service_logs directory is empty
            if not any(service_logs_dir.iterdir()):
                raise RuntimeError("No log source available: Loki is not available and service_logs directory is empty")
            
            logger.info("Using service_logs directory for logs")
        
        logger.info(f"Initializing ContextBuilder with log source {self.log_source}")
    
    def get_task_description(self, issue_id: str) -> str:
        """
        Get a description of the task for this agent.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Task description string
        """
        return (f"Gather and analyze all relevant logs, metrics, and system information for issue {issue_id}. "
                f"Create a comprehensive context to help identify the root cause of the problem.")
    
    async def collect_logs(self, issue_id: str, time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Collect relevant logs for the given issue.
        
        Args:
            issue_id: Issue ID to collect logs for
            time_window_minutes: Time window in minutes to look back
            
        Returns:
            List of log entries
            
        Raises:
            RuntimeError: If no logs are available from either Loki or service_logs directory
        """
        logger.info(f"Collecting logs for issue {issue_id} from {self.log_source}")
        
        logs = []
        error_messages = []
        
        if self.log_source == "loki" and self.loki_client:
            try:
                # First get error logs
                error_logs = await self.loki_client.get_error_logs(time_window_minutes)
                
                # If we have error logs, get logs from the affected services
                if error_logs:
                    affected_services = set(log["service"] for log in error_logs)
                    all_logs = error_logs
                    
                    # Get all logs from affected services
                    for service in affected_services:
                        service_logs = await self.loki_client.get_service_logs(
                            service, time_window_minutes
                        )
                        all_logs.extend(service_logs)
                    
                    # Remove duplicates and sort by timestamp
                    unique_logs = {log["timestamp"]: log for log in all_logs}.values()
                    logs = sorted(unique_logs, key=lambda x: x["timestamp"])
                    
                    logger.info(f"Collected {len(logs)} log entries from Loki")
                    return list(logs)
                else:
                    error_messages.append("No error logs found in Loki")
                    
            except Exception as e:
                error_messages.append(f"Error collecting logs from Loki: {str(e)}")
                logger.error(f"Error collecting logs from Loki: {e}")
                
        # Check for service_logs directory
        service_logs_dir = Path("data/logs/service_logs")
        if service_logs_dir.exists():
            try:
                all_logs = []
                # Read all JSON files in the service_logs directory
                for log_file in service_logs_dir.glob("*.json"):
                    with open(log_file, 'r') as f:
                        try:
                            file_logs = json.load(f)
                            if not isinstance(file_logs, list):
                                error_messages.append(f"Invalid log format in {log_file}: not a list")
                                continue
                            if not file_logs:
                                error_messages.append(f"Empty log file: {log_file}")
                                continue
                            all_logs.extend(file_logs)
                        except json.JSONDecodeError as e:
                            error_messages.append(f"Error parsing {log_file}: {str(e)}")
                            continue
                
                if all_logs:
                    # Filter logs within the time window
                    end_time = datetime.now()
                    start_time = end_time - timedelta(minutes=time_window_minutes)
                    
                    logs = [
                        log for log in all_logs
                        if start_time <= datetime.fromisoformat(log["timestamp"]) <= end_time
                    ]
                    
                    if logs:
                        logger.info(f"Collected {len(logs)} log entries from service_logs")
                        return logs
                    else:
                        error_messages.append("No logs found within the specified time window in service_logs")
                else:
                    error_messages.append("No valid logs found in service_logs directory")
                    
            except Exception as e:
                error_messages.append(f"Error reading service_logs: {str(e)}")
                logger.error(f"Error reading service_logs: {e}")
        else:
            error_messages.append("Service logs directory does not exist")
                
        # If we get here, no logs were available
        error_message = "No logs available. Errors encountered:\n" + "\n".join(error_messages)
        logger.error(error_message)
        raise RuntimeError(error_message)
    
    def collect_metrics(self, issue_id: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Collect relevant metrics for the given issue.
        
        Args:
            issue_id: Issue ID to collect metrics for
            time_window_minutes: Time window in minutes to look back
            
        Returns:
            Dictionary of metrics data
        """
        logger.info(f"Collecting metrics for issue {issue_id} from {self.metrics_source}")
        
        # This would be replaced with actual metrics collection from the configured source
        # For now, generate synthetic metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        # Generate time points
        timestamps = [start_time + timedelta(minutes=i) for i in range(time_window_minutes + 1)]
        
        # Generate metrics with an issue appearing at about 70% through the timeline
        issue_start_idx = int(len(timestamps) * 0.7)
        
        cpu_usage = []
        memory_usage = []
        request_count = []
        error_rate = []
        response_time = []
        
        for i in range(len(timestamps)):
            # CPU usage - rises during issue
            if i < issue_start_idx:
                cpu_usage.append(np.random.uniform(20, 40))
            else:
                cpu_usage.append(np.random.uniform(70, 95))
            
            # Memory usage - rises during issue
            if i < issue_start_idx:
                memory_usage.append(np.random.uniform(30, 50))
            else:
                memory_usage.append(np.random.uniform(60, 80))
            
            # Request count - drops slightly during issue
            if i < issue_start_idx:
                request_count.append(np.random.normal(1000, 50))
            else:
                request_count.append(np.random.normal(800, 100))
            
            # Error rate - spikes during issue
            if i < issue_start_idx:
                error_rate.append(np.random.beta(1, 20))  # Low error rate
            else:
                error_rate.append(np.random.beta(5, 10))  # Higher error rate
            
            # Response time - increases during issue
            if i < issue_start_idx:
                response_time.append(np.random.gamma(2, 0.1))
            else:
                response_time.append(np.random.gamma(5, 0.2))
        
        metrics = {
            "timestamps": [t.isoformat() for t in timestamps],
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "request_count": request_count,
            "error_rate": error_rate,
            "response_time": response_time
        }
        
        logger.info(f"Collected metrics with {len(timestamps)} data points")
        return metrics
    
    def collect_traces(self, issue_id: str, time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Collect execution traces related to the issue.
        
        Args:
            issue_id: Issue ID to collect traces for
            time_window_minutes: Time window in minutes to look back
            
        Returns:
            List of execution traces
        """
        logger.info(f"Collecting traces for issue {issue_id} from {self.traces_source}")
        
        # This would be replaced with actual trace collection from the configured source
        # For now, generate synthetic trace data
        traces = []
        
        # Generate some normal traces
        for i in range(5):
            trace = {
                "trace_id": f"trace-{issue_id}-{i}",
                "spans": [
                    {
                        "span_id": f"span-{i}-1",
                        "service": "api-gateway",
                        "operation": "process_request",
                        "duration_ms": np.random.uniform(10, 30),
                        "status": "ok"
                    },
                    {
                        "span_id": f"span-{i}-2",
                        "service": "auth-service",
                        "operation": "validate_token",
                        "duration_ms": np.random.uniform(50, 100),
                        "status": "ok"
                    },
                    {
                        "span_id": f"span-{i}-3",
                        "service": "user-service",
                        "operation": "get_user_profile",
                        "duration_ms": np.random.uniform(20, 50),
                        "status": "ok"
                    }
                ]
            }
            traces.append(trace)
        
        # Generate error trace
        error_trace = {
            "trace_id": f"trace-{issue_id}-error",
            "spans": [
                {
                    "span_id": "span-error-1",
                    "service": "api-gateway",
                    "operation": "process_request",
                    "duration_ms": np.random.uniform(10, 30),
                    "status": "ok"
                },
                {
                    "span_id": "span-error-2",
                    "service": "auth-service",
                    "operation": "validate_token",
                    "duration_ms": np.random.uniform(200, 300),  # Slow
                    "status": "error",
                    "error": "Token validation failure"
                },
                {
                    "span_id": "span-error-3",
                    "service": "user-service",
                    "operation": "get_user_profile",
                    "duration_ms": 0,  # Not reached
                    "status": "unknown"
                }
            ]
        }
        traces.append(error_trace)
        
        logger.info(f"Collected {len(traces)} traces")
        return traces
    
    def collect_related_issues(self, issue_id: str) -> List[Dict[str, Any]]:
        """
        Find past issues that may be related to the current one.
        
        Args:
            issue_id: Current issue ID
            
        Returns:
            List of related past issues
        """
        logger.info(f"Finding related issues for {issue_id}")
        
        # This would use a vector database to find similar issues
        # For now, generate synthetic data
        related_issues = [
            {
                "issue_id": "DEV-5123",
                "title": "Auth service token validation failures",
                "description": "Users experiencing authentication failures due to token validation issues",
                "resolution": "Fixed token expiration handling logic",
                "similarity_score": 0.92
            },
            {
                "issue_id": "DEV-3827",
                "title": "Intermittent auth timeouts",
                "description": "Authentication service timing out during peak load periods",
                "resolution": "Increased connection pool size and timeout settings",
                "similarity_score": 0.78
            },
            {
                "issue_id": "DEV-4291",
                "title": "Memory leak in auth service",
                "description": "Gradual memory increase leading to OOM errors",
                "resolution": "Fixed token cache implementation to properly evict expired entries",
                "similarity_score": 0.62
            }
        ]
        
        logger.info(f"Found {len(related_issues)} related issues")
        return related_issues
    
    def build_context(self, issue_id: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Build a comprehensive context for debugging the given issue.
        
        Args:
            issue_id: Issue ID to debug
            time_window_minutes: Time window in minutes to look back
            
        Returns:
            Dictionary containing all relevant context information
        """
        logger.info(f"Building context for issue {issue_id}")
        
        # Collect all relevant data
        logs = self.collect_logs(issue_id, time_window_minutes)
        metrics = self.collect_metrics(issue_id, time_window_minutes)
        traces = self.collect_traces(issue_id, time_window_minutes)
        related_issues = self.collect_related_issues(issue_id)
        
        # Build the context dictionary
        context = {
            "issue_id": issue_id,
            "timestamp": datetime.now().isoformat(),
            "time_window_minutes": time_window_minutes,
            "logs": logs,
            "metrics": metrics,
            "traces": traces,
            "related_issues": related_issues,
            "summary": self._generate_summary(logs, metrics, traces, related_issues)
        }
        
        # Save context to file
        self._save_context(context, issue_id)
        
        logger.info(f"Context built successfully for issue {issue_id}")
        return context
    
    def _generate_summary(self, logs: List[Dict[str, Any]], metrics: Dict[str, Any], 
                         traces: List[Dict[str, Any]], related_issues: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the collected data.
        
        Args:
            logs: Collected logs
            metrics: Collected metrics
            traces: Collected traces
            related_issues: Related past issues
            
        Returns:
            Summary string
        """
        # Count error logs
        error_logs = [log for log in logs if log["level"] in ["ERROR", "WARN"]]
        error_count = len(error_logs)
        
        # Check for spikes in metrics
        has_cpu_spike = any(cpu > 90 for cpu in metrics["cpu_usage"])
        has_memory_spike = any(mem > 85 for mem in metrics["memory_usage"])
        has_error_spike = any(err > 0.1 for err in metrics["error_rate"])
        
        # Check for failed traces
        failed_traces = [trace for trace in traces 
                         if any(span["status"] == "error" for span in trace["spans"])]
        
        # Generate summary text
        summary_parts = []
        
        if error_count > 0:
            summary_parts.append(f"Found {error_count} error/warning logs.")
        
        if has_cpu_spike:
            summary_parts.append("CPU usage spikes detected.")
        
        if has_memory_spike:
            summary_parts.append("Memory usage spikes detected.")
        
        if has_error_spike:
            summary_parts.append("Error rate spikes detected.")
        
        if failed_traces:
            summary_parts.append(f"Found {len(failed_traces)} failed execution traces.")
        
        if related_issues:
            most_similar = related_issues[0]
            summary_parts.append(f"Most similar past issue: {most_similar['issue_id']} - {most_similar['title']} "
                               f"(similarity: {most_similar['similarity_score']:.2f})")
        
        if summary_parts:
            summary = "Summary: " + " ".join(summary_parts)
        else:
            summary = "Summary: No significant issues detected in the collected data."
        
        return summary
    
    def _save_context(self, context: Dict[str, Any], issue_id: str) -> None:
        """
        Save the context to a file.
        
        Args:
            context: Context dictionary
            issue_id: Issue ID
        """
        try:
            # Create contexts directory if it doesn't exist
            contexts_dir = os.path.join('data', 'contexts')
            os.makedirs(contexts_dir, exist_ok=True)
            
            # Save context to JSON file
            filename = f"{issue_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(contexts_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(context, f, indent=2)
            
            logger.info(f"Context saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving context: {str(e)}") 