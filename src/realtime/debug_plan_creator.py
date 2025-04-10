"""
Debug Plan Creator - Creates debugging plans.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class DebugPlanCreator:
    """
    Agent that creates a plan for debugging based on the context gathered.
    """
    
    def __init__(self):
        """Initialize the DebugPlanCreator agent."""
        logger.info("Initializing DebugPlanCreator")
    
    def get_task_description(self, issue_id: str) -> str:
        """
        Get a description of the task for this agent.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Task description string
        """
        return (f"Analyze the context for issue {issue_id} and create a comprehensive debugging plan. "
                f"Identify key areas to investigate and actions to take to resolve the issue.")
    
    def create_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a debugging plan based on the context.
        
        Args:
            context: Context dictionary from ContextBuilder
            
        Returns:
            Dictionary containing the debugging plan
        """
        issue_id = context.get("issue_id", "unknown")
        logger.info(f"Creating debugging plan for issue {issue_id}")
        
        # Extract relevant information from context
        logs = context.get("logs", [])
        metrics = context.get("metrics", {})
        traces = context.get("traces", [])
        related_issues = context.get("related_issues", [])
        
        # Analyze error logs
        error_logs = [log for log in logs if log.get("level") in ["ERROR", "WARN"]]
        
        # Analyze metrics for anomalies
        metric_anomalies = self._find_metric_anomalies(metrics)
        
        # Analyze traces for bottlenecks or errors
        trace_issues = self._analyze_traces(traces)
        
        # Create plan steps based on findings
        plan_steps = []
        
        # Add steps based on log analysis
        if error_logs:
            log_step = {
                "step_id": 1,
                "title": "Investigate error logs",
                "description": f"Analyze {len(error_logs)} error/warning logs to identify patterns",
                "actions": [
                    "Review error messages and stack traces",
                    "Look for temporal patterns in errors",
                    "Check for correlation with system metrics"
                ],
                "estimated_time_minutes": 30
            }
            plan_steps.append(log_step)
        
        # Add steps based on metric anomalies
        if metric_anomalies:
            metric_step = {
                "step_id": len(plan_steps) + 1,
                "title": "Analyze system metrics",
                "description": f"Investigate {len(metric_anomalies)} metric anomalies",
                "actions": [f"Check {anomaly['metric']} ({anomaly['description']})" for anomaly in metric_anomalies],
                "estimated_time_minutes": 45
            }
            plan_steps.append(metric_step)
        
        # Add steps based on trace analysis
        if trace_issues:
            trace_step = {
                "step_id": len(plan_steps) + 1,
                "title": "Investigate service bottlenecks",
                "description": "Analyze distributed tracing to identify performance issues",
                "actions": [issue for issue in trace_issues],
                "estimated_time_minutes": 30
            }
            plan_steps.append(trace_step)
        
        # Add steps based on related issues
        if related_issues:
            similar_issue = related_issues[0]
            related_step = {
                "step_id": len(plan_steps) + 1,
                "title": "Check previous similar issues",
                "description": f"Review similar issue {similar_issue['issue_id']}: {similar_issue['title']}",
                "actions": [
                    f"Check if resolution applies: {similar_issue['resolution']}",
                    "Compare system behavior with previous incident",
                    "Check if same components are affected"
                ],
                "estimated_time_minutes": 20
            }
            plan_steps.append(related_step)
        
        # Add a generic investigation step if not enough specific steps
        if len(plan_steps) < 2:
            generic_step = {
                "step_id": len(plan_steps) + 1,
                "title": "General system investigation",
                "description": "Perform general investigation of system components",
                "actions": [
                    "Check auth service configuration",
                    "Verify database connections",
                    "Check network latency between services",
                    "Review recent deployments or changes"
                ],
                "estimated_time_minutes": 60
            }
            plan_steps.append(generic_step)
        
        # Add verification step
        verification_step = {
            "step_id": len(plan_steps) + 1,
            "title": "Verify issue resolution",
            "description": "Confirm that the issue has been resolved",
            "actions": [
                "Check logs for continued errors",
                "Monitor system metrics",
                "Perform test transactions",
                "Verify with users or monitors"
            ],
            "estimated_time_minutes": 15
        }
        plan_steps.append(verification_step)
        
        # Calculate total estimated time
        total_time = sum(step.get("estimated_time_minutes", 0) for step in plan_steps)
        
        # Create the plan
        plan = {
            "issue_id": issue_id,
            "plan_id": f"plan-{issue_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "total_estimated_time_minutes": total_time,
            "steps": plan_steps,
            "priority": self._determine_priority(context),
            "suspected_root_cause": self._identify_suspected_root_cause(context)
        }
        
        # Save the plan to file
        self._save_plan(plan, issue_id)
        
        logger.info(f"Created debugging plan with {len(plan_steps)} steps")
        return plan
    
    def _find_metric_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find anomalies in system metrics.
        
        Args:
            metrics: Dictionary of metrics data
            
        Returns:
            List of metric anomalies
        """
        anomalies = []
        
        # Check for CPU spikes
        cpu_values = metrics.get("cpu_usage", [])
        if any(cpu > 90 for cpu in cpu_values):
            anomalies.append({
                "metric": "CPU usage",
                "description": "High CPU utilization detected",
                "severity": "high"
            })
        
        # Check for memory usage
        memory_values = metrics.get("memory_usage", [])
        if any(mem > 85 for mem in memory_values):
            anomalies.append({
                "metric": "Memory usage",
                "description": "High memory utilization detected",
                "severity": "high" if any(mem > 95 for mem in memory_values) else "medium"
            })
        
        # Check for error rate spikes
        error_values = metrics.get("error_rate", [])
        if any(err > 0.1 for err in error_values):
            anomalies.append({
                "metric": "Error rate",
                "description": "Elevated error rate detected",
                "severity": "high" if any(err > 0.3 for err in error_values) else "medium"
            })
        
        # Check for response time increases
        response_values = metrics.get("response_time", [])
        if any(resp > 1.0 for resp in response_values):
            anomalies.append({
                "metric": "Response time",
                "description": "Slow response times detected",
                "severity": "medium"
            })
        
        return anomalies
    
    def _analyze_traces(self, traces: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze execution traces to find bottlenecks or errors.
        
        Args:
            traces: List of execution traces
            
        Returns:
            List of trace issues/actions
        """
        issues = []
        
        # Check for error spans
        error_spans = []
        for trace in traces:
            for span in trace.get("spans", []):
                if span.get("status") == "error":
                    error_spans.append({
                        "trace_id": trace.get("trace_id"),
                        "span_id": span.get("span_id"),
                        "service": span.get("service"),
                        "operation": span.get("operation"),
                        "error": span.get("error")
                    })
        
        if error_spans:
            for span in error_spans:
                issues.append(
                    f"Investigate {span['service']} error in operation {span['operation']}: {span.get('error', 'unknown error')}"
                )
        
        # Check for slow spans
        slow_spans = []
        for trace in traces:
            for span in trace.get("spans", []):
                if span.get("duration_ms", 0) > 150:  # Threshold for "slow"
                    slow_spans.append({
                        "trace_id": trace.get("trace_id"),
                        "span_id": span.get("span_id"),
                        "service": span.get("service"),
                        "operation": span.get("operation"),
                        "duration_ms": span.get("duration_ms")
                    })
        
        if slow_spans:
            for span in slow_spans:
                issues.append(
                    f"Check performance of {span['service']} in operation {span['operation']} ({span['duration_ms']}ms)"
                )
        
        return issues
    
    def _determine_priority(self, context: Dict[str, Any]) -> str:
        """
        Determine the priority of the issue based on context.
        
        Args:
            context: Context dictionary
            
        Returns:
            Priority level (critical, high, medium, low)
        """
        # Count error logs
        logs = context.get("logs", [])
        error_count = len([log for log in logs if log.get("level") in ["ERROR", "WARN"]])
        
        # Check metrics
        metrics = context.get("metrics", {})
        cpu_values = metrics.get("cpu_usage", [])
        memory_values = metrics.get("memory_usage", [])
        error_values = metrics.get("error_rate", [])
        
        # Check traces
        traces = context.get("traces", [])
        has_failed_traces = any(
            any(span.get("status") == "error" for span in trace.get("spans", []))
            for trace in traces
        )
        
        # Determine priority
        if (error_count > 20 or 
            any(cpu > 95 for cpu in cpu_values) or 
            any(mem > 95 for mem in memory_values) or
            any(err > 0.5 for err in error_values)):
            return "critical"
        elif (error_count > 10 or 
              any(cpu > 85 for cpu in cpu_values) or 
              any(mem > 85 for mem in memory_values) or
              any(err > 0.2 for err in error_values) or
              has_failed_traces):
            return "high"
        elif (error_count > 5 or 
              any(cpu > 70 for cpu in cpu_values) or 
              any(mem > 70 for mem in memory_values) or
              any(err > 0.1 for err in error_values)):
            return "medium"
        else:
            return "low"
    
    def _identify_suspected_root_cause(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify the suspected root cause based on context.
        
        Args:
            context: Context dictionary
            
        Returns:
            Dictionary with suspected root cause information
        """
        # Simple heuristic root cause analysis based on context
        logs = context.get("logs", [])
        metrics = context.get("metrics", {})
        traces = context.get("traces", [])
        
        # Check for auth service errors in logs
        auth_errors = [log for log in logs 
                      if log.get("service") == "auth-service" and log.get("level") in ["ERROR", "WARN"]]
        
        # Check for high CPU/memory
        cpu_values = metrics.get("cpu_usage", [])
        memory_values = metrics.get("memory_usage", [])
        has_high_cpu = any(cpu > 90 for cpu in cpu_values)
        has_high_memory = any(mem > 90 for mem in memory_values)
        
        # Check for auth service errors in traces
        auth_trace_errors = [
            span for trace in traces 
            for span in trace.get("spans", [])
            if span.get("service") == "auth-service" and span.get("status") == "error"
        ]
        
        # Determine likely root cause
        if auth_errors and auth_trace_errors:
            return {
                "service": "auth-service",
                "type": "service_error",
                "description": "Authentication service errors related to token validation",
                "confidence": "high"
            }
        elif has_high_cpu and has_high_memory and auth_errors:
            return {
                "service": "auth-service",
                "type": "resource_contention",
                "description": "Authentication service experiencing resource contention",
                "confidence": "medium"
            }
        elif has_high_memory:
            return {
                "service": "unknown",
                "type": "memory_leak",
                "description": "Possible memory leak in service",
                "confidence": "medium"
            }
        elif has_high_cpu:
            return {
                "service": "unknown",
                "type": "high_load",
                "description": "System under high load or possible infinite loop",
                "confidence": "medium"
            }
        else:
            return {
                "service": "unknown",
                "type": "unknown",
                "description": "Unable to determine root cause from available information",
                "confidence": "low"
            }
    
    def _save_plan(self, plan: Dict[str, Any], issue_id: str) -> None:
        """
        Save the debugging plan to a file.
        
        Args:
            plan: Debugging plan dictionary
            issue_id: Issue ID
        """
        try:
            # Create plans directory if it doesn't exist
            plans_dir = os.path.join('data', 'plans')
            os.makedirs(plans_dir, exist_ok=True)
            
            # Save plan to JSON file
            import json
            filename = f"{issue_id}_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(plans_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(plan, f, indent=2)
            
            logger.info(f"Debugging plan saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving debugging plan: {str(e)}") 