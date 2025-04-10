import json
import os
import datetime

# Get the path to the project root (one level up from scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Path to the log file
log_file = os.path.join(project_root, "data", "logs", "test_logs.json")

# Check if the file exists
if not os.path.exists(log_file):
    print(f"Error: Log file not found at {log_file}")
    print("Please run generate_logs.py first")
    exit(1)

# Load the logs
with open(log_file, "r") as f:
    logs = json.load(f)

# Print some stats
error_logs = [log for log in logs if log["level"] == "ERROR"]
warn_logs = [log for log in logs if log["level"] == "WARN"]
info_logs = [log for log in logs if log["level"] == "INFO"]
debug_logs = [log for log in logs if log["level"] == "DEBUG"]

print(f"Total logs: {len(logs)}")
print(f"Error logs: {len(error_logs)}")
print(f"Warning logs: {len(warn_logs)}")
print(f"Info logs: {len(info_logs)}")
print(f"Debug logs: {len(debug_logs)}")

# Print the most recent error
if error_logs:
    # Sort by timestamp (newest first)
    error_logs.sort(key=lambda x: x["timestamp"], reverse=True)
    newest_error = error_logs[0]
    print("\nMost recent error:")
    print(f"Time: {newest_error['timestamp']}")
    print(f"Service: {newest_error['service']}")
    print(f"Message: {newest_error['message']}") 