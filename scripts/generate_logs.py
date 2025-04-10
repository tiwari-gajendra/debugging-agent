import json
import random
import datetime
import os

# Get the path to the project root (one level up from scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Ensure the logs directory exists
logs_dir = os.path.join(project_root, "data", "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate 1000 random logs
logs = [
    {
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 60))).isoformat(),
        "level": random.choice(["INFO", "WARN", "ERROR", "DEBUG"]),
        "service": random.choice(["auth-service", "api-gateway", "user-service", "payment-service"]),
        "message": f"Log message {i}"
    } for i in range(1000)
]

# Generate some specific error logs for auth service
error_logs = [
    {
        "timestamp": (datetime.datetime.now() - datetime.timedelta(minutes=random.randint(30, 40))).isoformat(),
        "level": "ERROR",
        "service": "auth-service",
        "message": "Authentication failure: token validation error (ref: TEST-123)"
    } for _ in range(20)
]

# Add errors to the logs
logs.extend(error_logs)

# Save to file
output_file = os.path.join(logs_dir, "test_logs.json")
with open(output_file, "w") as f:
    json.dump(logs, f, indent=2)

print(f"Generated {len(logs)} logs with {len(error_logs)} errors in {output_file}") 