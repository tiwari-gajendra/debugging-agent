"""
Executor - Executes debugging plans.
"""

import os
import json
import logging
import time
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class Executor:
    """
    Agent that executes the debugging plan, performing diagnostic actions
    and capturing results.
    """
    
    def __init__(self, 
                aws_enabled: bool = False,
                aws_region: Optional[str] = None):
        """
        Initialize the Executor agent.
        
        Args:
            aws_enabled: Whether AWS integrations are enabled
            aws_region: AWS region for AWS operations
        """
        self.aws_enabled = aws_enabled
        self.aws_region = aws_region or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.execution_results = {}
        logger.info(f"Initializing Executor (AWS enabled: {aws_enabled})")
    
    def get_task_description(self, issue_id: str) -> str:
        """
        Get a description of the task for this agent.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Task description string
        """
        return (f"Execute the debugging plan for issue {issue_id}, performing all diagnostic actions "
                f"and collecting the results of each step.")
    
    def execute_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a debugging plan and capture results.
        
        Args:
            plan: Debugging plan from DebugPlanCreator
            context: Context from ContextBuilder
            
        Returns:
            Dictionary with execution results
        """
        issue_id = plan.get("issue_id", "unknown")
        logger.info(f"Executing debugging plan for issue {issue_id}")
        
        steps = plan.get("steps", [])
        if not steps:
            logger.warning("No steps in debugging plan to execute")
            return {"issue_id": issue_id, "results": []}
        
        results = []
        
        # Execute each step in the plan
        for step in steps:
            step_id = step.get("step_id")
            step_title = step.get("title", f"Step {step_id}")
            logger.info(f"Executing {step_title} (Step {step_id})")
            
            # Execute actions in this step
            action_results = []
            for action in step.get("actions", []):
                action_result = self._execute_action(action, context)
                action_results.append(action_result)
                
                # Simulate some time passing for the action
                time.sleep(0.5)  # For demonstration only
            
            # Compile results for this step
            step_result = {
                "step_id": step_id,
                "title": step_title,
                "status": "completed",
                "execution_time": datetime.now().isoformat(),
                "action_results": action_results
            }
            results.append(step_result)
            
            # For a real system, we might want to check if issues were found
            # and potentially stop execution if a critical problem is solved
        
        # Compile overall execution results
        execution_result = {
            "issue_id": issue_id,
            "plan_id": plan.get("plan_id"),
            "execution_id": f"exec-{issue_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "status": "completed",
            "results": results
        }
        
        # Save result to file
        self._save_execution_result(execution_result, issue_id)
        
        # Store locally for reference
        self.execution_results[issue_id] = execution_result
        
        logger.info(f"Completed execution of debugging plan for issue {issue_id}")
        return execution_result
    
    def _execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific debugging action.
        
        Args:
            action: Description of the action to perform
            context: Context information for the issue
            
        Returns:
            Dictionary with action result
        """
        logger.info(f"Executing action: {action}")
        
        # This would be replaced with actual action execution logic
        # For demonstration, we'll simulate different types of actions
        
        if "error logs" in action.lower():
            return self._check_error_logs(context)
        elif "metrics" in action.lower() or "cpu" in action.lower() or "memory" in action.lower():
            return self._check_system_metrics(action, context)
        elif "performance" in action.lower() or "bottleneck" in action.lower():
            return self._check_performance(action, context)
        elif "configuration" in action.lower() or "config" in action.lower():
            return self._check_configuration(action)
        elif "database" in action.lower() or "db" in action.lower():
            return self._check_database_connection()
        elif "network" in action.lower():
            return self._check_network_connectivity()
        elif "verify" in action.lower() or "monitor" in action.lower():
            return self._verify_system_status()
        else:
            # Generic action handling
            return {
                "action": action,
                "status": "completed",
                "findings": ["No specific findings for this action"],
                "execution_time": datetime.now().isoformat()
            }
    
    def _check_error_logs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze error logs from the context.
        
        Args:
            context: Context information with logs
            
        Returns:
            Dictionary with log analysis results
        """
        logs = context.get("logs", [])
        error_logs = [log for log in logs if log.get("level") in ["ERROR", "WARN"]]
        
        # Group errors by service
        services = {}
        for log in error_logs:
            service = log.get("service", "unknown")
            if service not in services:
                services[service] = []
            services[service].append(log)
        
        # Analyze patterns
        findings = []
        for service, logs in services.items():
            findings.append(f"Found {len(logs)} errors in {service}")
            
            # Look for common error messages
            error_messages = {}
            for log in logs:
                msg = log.get("message", "")
                if msg in error_messages:
                    error_messages[msg] += 1
                else:
                    error_messages[msg] = 1
            
            # Report common errors
            for msg, count in error_messages.items():
                if count > 1:  # If it appears multiple times
                    findings.append(f"  - Repeated error ({count}x): {msg[:80]}...")
        
        if not findings:
            findings = ["No significant error patterns found in logs"]
        
        return {
            "action": "Analyze error logs",
            "status": "completed",
            "findings": findings,
            "error_count": len(error_logs),
            "execution_time": datetime.now().isoformat()
        }
    
    def _check_system_metrics(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze system metrics from the context.
        
        Args:
            action: Description of the metric to check
            context: Context information with metrics
            
        Returns:
            Dictionary with metric analysis results
        """
        metrics = context.get("metrics", {})
        
        findings = []
        
        # Check CPU usage
        if "cpu" in action.lower():
            cpu_values = metrics.get("cpu_usage", [])
            if cpu_values:
                avg_cpu = sum(cpu_values) / len(cpu_values)
                max_cpu = max(cpu_values)
                findings.append(f"CPU usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%")
                
                if max_cpu > 90:
                    findings.append("  - CRITICAL: CPU utilization exceeded 90%")
                elif max_cpu > 75:
                    findings.append("  - WARNING: CPU utilization exceeded 75%")
        
        # Check memory usage
        if "memory" in action.lower():
            memory_values = metrics.get("memory_usage", [])
            if memory_values:
                avg_memory = sum(memory_values) / len(memory_values)
                max_memory = max(memory_values)
                findings.append(f"Memory usage: avg={avg_memory:.1f}%, max={max_memory:.1f}%")
                
                if max_memory > 90:
                    findings.append("  - CRITICAL: Memory utilization exceeded 90%")
                elif max_memory > 75:
                    findings.append("  - WARNING: Memory utilization exceeded 75%")
        
        # Check error rate
        if "error" in action.lower():
            error_values = metrics.get("error_rate", [])
            if error_values:
                avg_error = sum(error_values) / len(error_values)
                max_error = max(error_values)
                findings.append(f"Error rate: avg={avg_error:.3f}, max={max_error:.3f}")
                
                if max_error > 0.3:
                    findings.append("  - CRITICAL: Error rate exceeded 30%")
                elif max_error > 0.1:
                    findings.append("  - WARNING: Error rate exceeded 10%")
        
        if not findings:
            findings = ["No significant metric issues found"]
        
        return {
            "action": "Analyze system metrics",
            "status": "completed",
            "findings": findings,
            "execution_time": datetime.now().isoformat()
        }
    
    def _check_performance(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance metrics and traces.
        
        Args:
            action: Description of the performance to check
            context: Context information with traces
            
        Returns:
            Dictionary with performance analysis results
        """
        traces = context.get("traces", [])
        
        findings = []
        
        # Extract the service to check from the action
        service_to_check = None
        if "auth" in action.lower() or "authentication" in action.lower():
            service_to_check = "auth-service"
        elif "api" in action.lower():
            service_to_check = "api-gateway"
        elif "user" in action.lower():
            service_to_check = "user-service"
        
        # Check for performance issues in traces
        slow_spans = []
        for trace in traces:
            for span in trace.get("spans", []):
                if service_to_check and span.get("service") != service_to_check:
                    continue
                    
                duration = span.get("duration_ms", 0)
                if duration > 200:  # Consider spans > 200ms as slow
                    slow_spans.append({
                        "service": span.get("service"),
                        "operation": span.get("operation"),
                        "duration_ms": duration
                    })
        
        if slow_spans:
            findings.append(f"Found {len(slow_spans)} slow operations")
            
            # Group by service and operation
            by_service_op = {}
            for span in slow_spans:
                key = f"{span['service']}.{span['operation']}"
                if key not in by_service_op:
                    by_service_op[key] = []
                by_service_op[key].append(span['duration_ms'])
            
            # Report averages
            for key, durations in by_service_op.items():
                avg_duration = sum(durations) / len(durations)
                findings.append(f"  - {key}: avg={avg_duration:.1f}ms across {len(durations)} spans")
        
        if not findings:
            findings = ["No significant performance issues found"]
        
        return {
            "action": "Check service performance",
            "status": "completed",
            "findings": findings,
            "slow_operations_count": len(slow_spans),
            "execution_time": datetime.now().isoformat()
        }
    
    def _check_configuration(self, action: str) -> Dict[str, Any]:
        """
        Check service configurations.
        
        Args:
            action: Description of the configuration to check
            
        Returns:
            Dictionary with configuration check results
        """
        # This would be replaced with actual configuration checks
        # For demonstration, we'll return simulated results
        
        service = "auth-service"
        if "auth" in action.lower():
            service = "auth-service"
        elif "api" in action.lower():
            service = "api-gateway"
        elif "user" in action.lower():
            service = "user-service"
        
        findings = [
            f"Checked {service} configuration",
            "All configuration parameters are within expected ranges",
            "Token expiration is set to 3600 seconds (recommended)",
            "Rate limiting is properly configured"
        ]
        
        return {
            "action": f"Check {service} configuration",
            "status": "completed",
            "findings": findings,
            "execution_time": datetime.now().isoformat()
        }
    
    def _check_database_connection(self) -> Dict[str, Any]:
        """
        Check database connectivity and performance.
        
        Returns:
            Dictionary with database check results
        """
        # This would be replaced with actual database checks
        # For demonstration, we'll return simulated results
        
        findings = [
            "Database connection successful",
            "Connection pool size: 20 (recommended: 30)",
            "Average query time: 25ms",
            "No deadlocks detected in the last hour",
            "Connection timeout is set to 5 seconds (recommended: 10 seconds)"
        ]
        
        return {
            "action": "Check database connections",
            "status": "completed",
            "findings": findings,
            "execution_time": datetime.now().isoformat()
        }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """
        Check network connectivity between services.
        
        Returns:
            Dictionary with network check results
        """
        # This would be replaced with actual network checks
        # For demonstration, we'll simulate checking network latency
        
        services = ["auth-service", "api-gateway", "user-service", "payment-service"]
        findings = ["Network connectivity check completed"]
        
        # Simulate latency checks
        for i, service1 in enumerate(services):
            for service2 in services[i+1:]:
                latency = round(10 + 20 * 0.5, 1)  # Simulated latency
                findings.append(f"  - {service1} → {service2}: {latency}ms")
        
        findings.append("All services are reachable")
        findings.append("No packet loss detected")
        
        return {
            "action": "Check network connectivity",
            "status": "completed",
            "findings": findings,
            "execution_time": datetime.now().isoformat()
        }
    
    def _verify_system_status(self) -> Dict[str, Any]:
        """
        Verify the system status after debugging actions.
        
        Returns:
            Dictionary with verification results
        """
        # This would be replaced with actual verification steps
        # For demonstration, we'll simulate checking system status
        
        findings = [
            "System status verification completed",
            "All services are operational",
            "Error rates have returned to normal levels",
            "Response times are within acceptable ranges",
            "No new error logs detected in the last 5 minutes"
        ]
        
        return {
            "action": "Verify system status",
            "status": "completed",
            "findings": findings,
            "execution_time": datetime.now().isoformat()
        }
    
    def _save_execution_result(self, result: Dict[str, Any], issue_id: str) -> None:
        """
        Save the execution result to a file.
        
        Args:
            result: Execution result dictionary
            issue_id: Issue ID
        """
        try:
            # Create executions directory if it doesn't exist
            executions_dir = os.path.join('data', 'executions')
            os.makedirs(executions_dir, exist_ok=True)
            
            # Save result to JSON file
            filename = f"{issue_id}_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(executions_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Execution result saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving execution result: {str(e)}") 