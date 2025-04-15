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
                log_source: str = "service_logs",  # Changed default to service_logs
                metrics_source: str = "prometheus",
                traces_source: str = "jaeger",
                vector_db_path: Optional[str] = None):
        """
        Initialize the ContextBuilder agent.
        
        Args:
            log_source: Source of logs (service_logs, loki, cloudwatch, etc.)
            metrics_source: Source of metrics (prometheus, cloudwatch, etc.)
            traces_source: Source of traces (jaeger, x-ray, etc.)
            vector_db_path: Path to the vector database for RAG
        """
        # Initialize data sources
        self.log_source = log_source
        self.metrics_source = metrics_source
        self.traces_source = traces_source
        
        # Set up context storage
        self.context_dir = Path("data/contexts")
        self.context_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize in-memory cache
        self.context_cache = {}
        
        # Initialize clients
        self.loki_client = None
        if log_source == "loki":
            try:
                self.loki_client = LokiClient()
                logger.info("Successfully initialized Loki client")
            except Exception as e:
                logger.warning(f"Failed to initialize Loki client: {e}. Falling back to service_logs.")
                self.log_source = "service_logs"
        
        # Initialize vector database if path is provided
        self.vector_db = None
        if vector_db_path:
            self._init_vector_db(vector_db_path)
            
        logger.info(f"ContextBuilder initialized with log source: {self.log_source}")
    
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
                            if isinstance(file_logs, list):
                                all_logs.extend(file_logs)
                            elif isinstance(file_logs, dict) and 'logs' in file_logs:
                                all_logs.extend(file_logs['logs'])
                            else:
                                error_messages.append(f"Invalid log format in {log_file}")
                        except json.JSONDecodeError as e:
                            error_messages.append(f"Error parsing {log_file}: {str(e)}")
                            continue
                
                if all_logs:
                    try:
                        end_time = datetime.now()
                        start_time = end_time - timedelta(minutes=time_window_minutes)
                        filtered_logs = [
                            log for log in all_logs
                            if 'timestamp' not in log or  # Include logs without timestamps
                            start_time <= datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00')) <= end_time
                        ]
                        logger.info(f"Collected {len(filtered_logs)} log entries from service_logs")
                        return filtered_logs
                    except Exception as e:
                        logger.warning(f"Error filtering logs by timestamp: {e}. Returning all logs.")
                        return all_logs
                else:
                    error_messages.append("No logs found in service_logs directory")
            except Exception as e:
                error_messages.append(f"Error reading service_logs: {str(e)}")
                logger.error(f"Error reading service logs: {e}")
        else:
            error_messages.append("Service logs directory does not exist")
            logger.warning("Service logs directory does not exist")
                
        # If we get here, no logs were available
        error_message = "No logs available. Errors encountered:\n" + "\n".join(error_messages)
        logger.error(error_message)
        return []  # Return empty list instead of raising exception
    
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
        
        try:
            timestamps = []
            cpu_usage = []
            memory_usage = []
            request_count = []
            error_rate = []
            response_time = []
            
            # Generate sample metrics data
            for i in range(time_window_minutes):
                timestamps.append(datetime.now() - timedelta(minutes=i))
                cpu_usage.append(np.random.uniform(20, 80))
                memory_usage.append(np.random.uniform(40, 90))
                request_count.append(int(np.random.uniform(100, 1000)))
                error_rate.append(np.random.uniform(0, 5))
                response_time.append(np.random.uniform(50, 200))
            
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
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}
    
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
                    "status": "not_executed"
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
            # Save context to JSON file
            filename = f"{issue_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.context_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(context, f, indent=2)
            
            logger.info(f"Context saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving context: {str(e)}")
    
    def get_context(self, issue_id: str) -> Dict[str, Any]:
        """Get existing context for an issue."""
        # First check cache
        if issue_id in self.context_cache:
            return self.context_cache[issue_id]
            
        # Then check file storage
        context_file = self.context_dir / f"{issue_id}.json"
        if context_file.exists():
            try:
                with open(context_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading context file: {e}")
                
        return {}
        
    def update_context(self, issue_id: str, context: Dict[str, Any]) -> None:
        """Update context for an issue."""
        # Update cache
        self.context_cache[issue_id] = context
        
        # Update file storage
        context_file = self.context_dir / f"{issue_id}.json"
        try:
            with open(context_file, 'w') as f:
                json.dump(context, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing context file: {e}")
            
    def get_context_history(self, issue_id: str) -> List[Dict[str, Any]]:
        """Get context history for an issue."""
        context = self.get_context(issue_id)
        return context.get('history', []) 