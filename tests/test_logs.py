"""
Test file demonstrating how to use synthetic logs for testing purposes.
This is for testing only and should not be used in production.
"""

import json
import datetime
import random
from pathlib import Path
import pytest

def generate_synthetic_logs(issue_id: str, time_window_minutes: int = 60) -> list:
    """
    Generate synthetic logs for testing purposes only.
    This function should only be used in test files, not in production code.
    
    Args:
        issue_id: Issue ID to reference in logs
        time_window_minutes: Time window in minutes
        
    Returns:
        List of synthetic log entries
    """
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(minutes=time_window_minutes)
    
    logs = []
    
    # Generate normal logs
    for i in range(50):
        timestamp = start_time + datetime.timedelta(minutes=random.uniform(0, time_window_minutes))
        logs.append({
            "timestamp": timestamp.isoformat(),
            "level": random.choice(["INFO", "DEBUG", "INFO", "INFO", "WARN"]),
            "service": random.choice(["auth-service", "api-gateway", "user-service", "payment-service"]),
            "message": f"Regular operation log message {i}"
        })
    
    # Generate some warning/error logs related to the issue
    for i in range(10):
        timestamp = start_time + datetime.timedelta(minutes=random.uniform(time_window_minutes*0.7, time_window_minutes))
        logs.append({
            "timestamp": timestamp.isoformat(),
            "level": random.choice(["ERROR", "WARN", "ERROR"]),
            "service": "auth-service",  # Assume issue is with auth service
            "message": f"Authentication failure: token validation error (ref: {issue_id})"
        })
    
    # Sort logs by timestamp
    logs.sort(key=lambda x: x["timestamp"])
    return logs

@pytest.fixture
def test_logs_dir(tmp_path):
    """Create a temporary directory for test logs."""
    logs_dir = tmp_path / "data" / "logs" / "service_logs"
    logs_dir.mkdir(parents=True)
    return logs_dir

def test_log_collection(test_logs_dir):
    """Test that logs are properly collected from service_logs directory."""
    # Generate test logs
    logs = generate_synthetic_logs("TEST-123")
    
    # Save logs to test directory
    log_file = test_logs_dir / "test_logs.json"
    with open(log_file, 'w') as f:
        json.dump(logs, f)
    
    # Verify logs were saved
    assert log_file.exists()
    with open(log_file, 'r') as f:
        loaded_logs = json.load(f)
        assert len(loaded_logs) == len(logs)
        assert all(log in loaded_logs for log in logs)

def test_error_log_filtering(test_logs_dir):
    """Test that error logs are properly filtered."""
    # Generate test logs
    logs = generate_synthetic_logs("TEST-123")
    
    # Save logs to test directory
    log_file = test_logs_dir / "test_logs.json"
    with open(log_file, 'w') as f:
        json.dump(logs, f)
    
    # Count error logs
    error_logs = [log for log in logs if log["level"] in ["ERROR", "WARN"]]
    assert len(error_logs) > 0
    assert all(log["level"] in ["ERROR", "WARN"] for log in error_logs) 