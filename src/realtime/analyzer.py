"""
Analyzer - Analyzes debugging results.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class Analyzer:
    """
    Agent that analyzes logs, metrics, traces, and system state
    to determine the root cause of an issue.
    """
    
    def __init__(self):
        """Initialize the Analyzer agent."""
        logger.info("Initializing Analyzer")
    
    def get_task_description(self, issue_id: str) -> str:
        """
        Get a description of the task for this agent.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Task description string
        """
        return (f"Analyze the execution results for issue {issue_id}, identify patterns, "
                f"determine the root cause, and recommend a solution.")
    
    def analyze(self, 
               context: Dict[str, Any],
               plan: Dict[str, Any],
               execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze debugging results to determine the root cause and solution.
        
        Args:
            context: Context information from ContextBuilder
            plan: Debugging plan from DebugPlanCreator
            execution_result: Results from Executor
            
        Returns:
            Dictionary with analysis results
        """
        issue_id = context.get("issue_id", "unknown")
        logger.info(f"Analyzing debugging results for issue {issue_id}")
        
        # Extract findings from execution results
        findings = self._extract_findings(execution_result)
        
        # Analyze error patterns
        error_analysis = self._analyze_errors(context, findings)
        
        # Analyze performance issues
        performance_analysis = self._analyze_performance(context, findings)
        
        # Analyze system resource issues
        resource_analysis = self._analyze_resources(context, findings)
        
        # Determine root cause
        root_cause, confidence = self._determine_root_cause(
            error_analysis, performance_analysis, resource_analysis, plan
        )
        
        # Generate solution recommendations
        solution = self._generate_solution(root_cause, context, findings)
        
        # Create visualization for the issue
        chart_path = self._create_visualization(context, root_cause, issue_id)
        
        # Compile analysis results
        analysis = {
            "issue_id": issue_id,
            "analysis_id": f"analysis-{issue_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "error_analysis": error_analysis,
            "performance_analysis": performance_analysis,
            "resource_analysis": resource_analysis,
            "root_cause": root_cause,
            "confidence": confidence,
            "solution": solution,
            "visualization_path": chart_path
        }
        
        # Save the analysis to file
        self._save_analysis(analysis, issue_id)
        
        logger.info(f"Analysis completed for issue {issue_id}")
        logger.info(f"Root cause: {root_cause.get('description')} (confidence: {confidence})")
        return analysis
    
    def _extract_findings(self, execution_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract findings from execution results.
        
        Args:
            execution_result: Results from Executor
            
        Returns:
            List of findings with metadata
        """
        all_findings = []
        
        # Process each step in the execution result
        for step in execution_result.get("results", []):
            step_title = step.get("title", "")
            step_id = step.get("step_id", "")
            
            # Process action results in this step
            for action_result in step.get("action_results", []):
                action = action_result.get("action", "")
                
                for finding in action_result.get("findings", []):
                    all_findings.append({
                        "step_id": step_id,
                        "step_title": step_title,
                        "action": action,
                        "finding": finding,
                        "metadata": {k: v for k, v in action_result.items() 
                                  if k not in ["action", "findings", "status"]}
                    })
        
        logger.info(f"Extracted {len(all_findings)} findings from execution results")
        return all_findings
    
    def _analyze_errors(self, context: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze error patterns from the context and findings.
        
        Args:
            context: Context information
            findings: Extracted findings from execution results
            
        Returns:
            Dictionary with error analysis results
        """
        # Extract error-related findings
        error_findings = [f for f in findings if "error" in f.get("action", "").lower() 
                         or "error" in f.get("finding", "").lower()]
        
        # Get error logs from context
        logs = context.get("logs", [])
        error_logs = [log for log in logs if log.get("level") in ["ERROR", "WARN"]]
        
        # Determine if specific services have more errors
        services = {}
        for log in error_logs:
            service = log.get("service", "unknown")
            if service not in services:
                services[service] = 0
            services[service] += 1
        
        # Sort services by error count
        service_errors = sorted(
            [(service, count) for service, count in services.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Try to identify common error patterns
        error_patterns = {}
        for log in error_logs:
            msg = log.get("message", "")
            # Simple pattern extraction (could be more sophisticated)
            pattern = " ".join(msg.split()[:5])  # First 5 words as pattern
            
            if pattern not in error_patterns:
                error_patterns[pattern] = 0
            error_patterns[pattern] += 1
        
        # Sort patterns by frequency
        sorted_patterns = sorted(
            [(pattern, count) for pattern, count in error_patterns.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "error_count": len(error_logs),
            "service_errors": service_errors[:3],  # Top 3 services by error count
            "common_patterns": sorted_patterns[:3],  # Top 3 error patterns
            "relevant_findings": [f.get("finding") for f in error_findings]
        }
    
    def _analyze_performance(self, context: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance issues from the context and findings.
        
        Args:
            context: Context information
            findings: Extracted findings from execution results
            
        Returns:
            Dictionary with performance analysis results
        """
        # Extract performance-related findings
        perf_findings = [f for f in findings if "performance" in f.get("action", "").lower() 
                        or "slow" in f.get("finding", "").lower()
                        or "latency" in f.get("finding", "").lower()]
        
        # Get trace information from context
        traces = context.get("traces", [])
        
        # Extract span durations by service
        service_durations = {}
        for trace in traces:
            for span in trace.get("spans", []):
                service = span.get("service", "unknown")
                duration = span.get("duration_ms", 0)
                
                if service not in service_durations:
                    service_durations[service] = []
                service_durations[service].append(duration)
        
        # Calculate average durations
        avg_durations = {}
        for service, durations in service_durations.items():
            if durations:  # Ensure non-empty list
                avg_durations[service] = sum(durations) / len(durations)
        
        # Sort services by average duration
        sorted_durations = sorted(
            [(service, avg) for service, avg in avg_durations.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check for performance outliers (services with unusually high latency)
        outliers = []
        all_durations = [d for durations in service_durations.values() for d in durations]
        if all_durations:
            avg_all = sum(all_durations) / len(all_durations)
            for service, durations in service_durations.items():
                if durations:
                    service_avg = sum(durations) / len(durations)
                    if service_avg > 2 * avg_all:  # Simple outlier detection
                        outliers.append({
                            "service": service,
                            "avg_duration": service_avg,
                            "overall_avg": avg_all,
                            "ratio": service_avg / avg_all
                        })
        
        return {
            "service_latencies": sorted_durations,
            "outliers": outliers,
            "relevant_findings": [f.get("finding") for f in perf_findings]
        }
    
    def _analyze_resources(self, context: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze system resource issues from the context and findings.
        
        Args:
            context: Context information
            findings: Extracted findings from execution results
            
        Returns:
            Dictionary with resource analysis results
        """
        # Extract resource-related findings
        resource_findings = [f for f in findings if "cpu" in f.get("action", "").lower() 
                            or "memory" in f.get("action", "").lower()
                            or "resource" in f.get("finding", "").lower()]
        
        # Get metrics information from context
        metrics = context.get("metrics", {})
        
        # Calculate resource statistics
        cpu_stats = self._calculate_stats(metrics.get("cpu_usage", []))
        memory_stats = self._calculate_stats(metrics.get("memory_usage", []))
        
        # Determine if resource issues exist
        has_cpu_issue = cpu_stats.get("max", 0) > 85
        has_memory_issue = memory_stats.get("max", 0) > 85
        
        # Check for correlation between resources and errors
        error_rates = metrics.get("error_rate", [])
        cpu_values = metrics.get("cpu_usage", [])
        memory_values = metrics.get("memory_usage", [])
        
        # Simple correlation check (not a true correlation coefficient)
        cpu_error_correlation = False
        memory_error_correlation = False
        
        if error_rates and cpu_values and len(error_rates) == len(cpu_values):
            # Check if high CPU coincides with high error rates
            high_cpu_indices = [i for i, cpu in enumerate(cpu_values) if cpu > 75]
            high_error_indices = [i for i, err in enumerate(error_rates) if err > 0.1]
            
            # Check overlap
            overlap = len(set(high_cpu_indices).intersection(high_error_indices))
            if overlap > 0 and high_error_indices:
                cpu_error_correlation = overlap / len(high_error_indices) > 0.5
        
        if error_rates and memory_values and len(error_rates) == len(memory_values):
            # Check if high memory coincides with high error rates
            high_mem_indices = [i for i, mem in enumerate(memory_values) if mem > 75]
            high_error_indices = [i for i, err in enumerate(error_rates) if err > 0.1]
            
            # Check overlap
            overlap = len(set(high_mem_indices).intersection(high_error_indices))
            if overlap > 0 and high_error_indices:
                memory_error_correlation = overlap / len(high_error_indices) > 0.5
        
        return {
            "cpu_stats": cpu_stats,
            "memory_stats": memory_stats,
            "has_cpu_issue": has_cpu_issue,
            "has_memory_issue": has_memory_issue,
            "cpu_error_correlation": cpu_error_correlation,
            "memory_error_correlation": memory_error_correlation,
            "relevant_findings": [f.get("finding") for f in resource_findings]
        }
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with statistics (min, max, avg)
        """
        if not values:
            return {"min": 0, "max": 0, "avg": 0}
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values)
        }
    
    def _determine_root_cause(self, 
                             error_analysis: Dict[str, Any],
                             performance_analysis: Dict[str, Any],
                             resource_analysis: Dict[str, Any],
                             plan: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        Determine the most likely root cause of the issue.
        
        Args:
            error_analysis: Error analysis results
            performance_analysis: Performance analysis results
            resource_analysis: Resource analysis results
            plan: Original debugging plan
            
        Returns:
            Tuple of (root cause dict, confidence level)
        """
        # Get suspected root cause from the plan
        suspected_cause = plan.get("suspected_root_cause", {})
        suspected_service = suspected_cause.get("service", "unknown")
        
        # Check for high error count in the suspected service
        service_errors = error_analysis.get("service_errors", [])
        service_with_most_errors = service_errors[0][0] if service_errors else "unknown"
        
        # Check for performance outliers
        outliers = performance_analysis.get("outliers", [])
        slow_services = [o.get("service") for o in outliers]
        
        # Check resource issues
        has_cpu_issue = resource_analysis.get("has_cpu_issue", False)
        has_memory_issue = resource_analysis.get("has_memory_issue", False)
        cpu_error_correlation = resource_analysis.get("cpu_error_correlation", False)
        memory_error_correlation = resource_analysis.get("memory_error_correlation", False)
        
        # Decision logic for root cause
        root_cause = {}
        confidence = "low"
        
        # Case 1: Errors, performance, and resources all point to the same service
        if (service_with_most_errors in slow_services and 
            (cpu_error_correlation or memory_error_correlation) and
            service_with_most_errors == suspected_service):
            
            # High confidence case
            root_cause = {
                "service": service_with_most_errors,
                "type": "combined_issue",
                "description": f"Multiple issues detected in {service_with_most_errors}: "
                             f"high error rate, slow performance, and resource contention"
            }
            confidence = "high"
        
        # Case 2: Clear performance issue with a service
        elif slow_services and len(outliers) > 0 and outliers[0].get("ratio", 1) > 3:
            slowest_service = outliers[0].get("service")
            root_cause = {
                "service": slowest_service,
                "type": "performance_issue",
                "description": f"Performance bottleneck in {slowest_service}, "
                             f"with latency {outliers[0].get('ratio', 0):.1f}x higher than average"
            }
            confidence = "high" if slowest_service == suspected_service else "medium"
        
        # Case 3: Resource issues with error correlation
        elif (has_cpu_issue or has_memory_issue) and (cpu_error_correlation or memory_error_correlation):
            resource_type = "CPU" if (has_cpu_issue and cpu_error_correlation) else "memory"
            root_cause = {
                "service": service_with_most_errors,
                "type": "resource_issue",
                "description": f"{resource_type} contention in {service_with_most_errors} "
                             f"causing elevated error rates"
            }
            confidence = "medium"
        
        # Case 4: Just high error rates
        elif service_errors and error_analysis.get("error_count", 0) > 10:
            root_cause = {
                "service": service_with_most_errors,
                "type": "error_spike",
                "description": f"Elevated error rate in {service_with_most_errors}, "
                             f"possibly due to application errors"
            }
            confidence = "medium" if service_with_most_errors == suspected_service else "low"
        
        # Case 5: Fall back to the suspected cause
        else:
            root_cause = suspected_cause
            confidence = suspected_cause.get("confidence", "low")
        
        return root_cause, confidence
    
    def _generate_solution(self, 
                          root_cause: Dict[str, Any], 
                          context: Dict[str, Any],
                          findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate solution recommendations based on the root cause.
        
        Args:
            root_cause: Identified root cause
            context: Context information
            findings: Execution findings
            
        Returns:
            Dictionary with solution recommendations
        """
        cause_type = root_cause.get("type", "unknown")
        service = root_cause.get("service", "unknown")
        
        # Check for related past issues
        related_issues = context.get("related_issues", [])
        past_solutions = [
            issue.get("resolution", "")
            for issue in related_issues
            if issue.get("title", "").lower().find(service.lower()) >= 0
        ]
        
        # Base recommendations on root cause type
        immediate_actions = []
        long_term_actions = []
        
        if cause_type == "performance_issue":
            immediate_actions = [
                f"Increase resources allocated to {service}",
                f"Check for blocking operations in {service}",
                "Verify database query performance",
                "Check for network latency between services"
            ]
            long_term_actions = [
                "Implement caching for frequently accessed data",
                "Optimize database queries",
                "Consider service refactoring to improve performance",
                "Implement circuit breakers for resilience"
            ]
        
        elif cause_type == "resource_issue":
            immediate_actions = [
                f"Scale up {service} instances",
                "Verify resource limits are properly set",
                "Check for resource leaks",
                "Monitor resource utilization"
            ]
            long_term_actions = [
                "Implement auto-scaling based on load",
                "Optimize resource usage",
                "Consider containerization for better resource isolation",
                "Implement resource monitoring and alerting"
            ]
        
        elif cause_type == "error_spike":
            immediate_actions = [
                f"Analyze error logs for {service}",
                "Check recent code changes",
                "Verify configuration settings",
                "Test service endpoints"
            ]
            long_term_actions = [
                "Implement comprehensive error handling",
                "Add more detailed logging",
                "Set up automated testing",
                "Create runbooks for common errors"
            ]
        
        elif cause_type == "combined_issue":
            immediate_actions = [
                f"Restart {service} service",
                "Check for memory leaks",
                "Increase resource allocation",
                "Verify external dependencies"
            ]
            long_term_actions = [
                "Conduct code review for performance issues",
                "Implement performance testing",
                "Set up monitoring and alerting",
                "Consider service decomposition"
            ]
        
        else:
            # Generic recommendations
            immediate_actions = [
                "Verify service configurations",
                "Check for recent changes",
                "Analyze logs for patterns",
                "Test service functionality"
            ]
            long_term_actions = [
                "Improve monitoring and alerting",
                "Implement automated testing",
                "Create runbooks for incident response",
                "Conduct regular system health checks"
            ]
        
        # Include past solutions if available
        if past_solutions:
            immediate_actions.append(f"Consider previous solution: {past_solutions[0]}")
        
        return {
            "immediate_actions": immediate_actions,
            "long_term_actions": long_term_actions,
            "past_solutions": past_solutions
        }
    
    def _create_visualization(self, 
                             context: Dict[str, Any], 
                             root_cause: Dict[str, Any],
                             issue_id: str) -> str:
        """
        Create a visualization of the issue.
        
        Args:
            context: Context information
            root_cause: Identified root cause
            issue_id: Issue ID
            
        Returns:
            Path to the generated visualization
        """
        try:
            # Create visualizations directory if it doesn't exist
            vis_dir = os.path.join('data', 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Extract metrics for visualization
            metrics = context.get("metrics", {})
            timestamps_str = metrics.get("timestamps", [])
            timestamps = [datetime.fromisoformat(ts) for ts in timestamps_str]
            
            # Convert timestamps to simple x-axis values
            if timestamps:
                x_values = range(len(timestamps))
            else:
                # If no timestamps, create dummy data
                x_values = range(10)
                metrics = {
                    "cpu_usage": [30, 35, 40, 45, 80, 90, 95, 85, 70, 60],
                    "memory_usage": [40, 42, 45, 50, 60, 75, 85, 80, 70, 65],
                    "error_rate": [0.01, 0.01, 0.02, 0.05, 0.1, 0.3, 0.4, 0.3, 0.1, 0.05]
                }
            
            # Create a figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Plot CPU usage
            cpu_values = metrics.get("cpu_usage", [])
            if cpu_values:
                axs[0].plot(x_values, cpu_values, 'b-', label='CPU Usage (%)')
                axs[0].set_ylabel('CPU Usage (%)')
                axs[0].set_title('System Metrics During Issue')
                axs[0].grid(True)
                axs[0].legend()
            
            # Plot Memory usage
            memory_values = metrics.get("memory_usage", [])
            if memory_values:
                axs[1].plot(x_values, memory_values, 'g-', label='Memory Usage (%)')
                axs[1].set_ylabel('Memory Usage (%)')
                axs[1].grid(True)
                axs[1].legend()
            
            # Plot Error rate
            error_values = metrics.get("error_rate", [])
            if error_values:
                axs[2].plot(x_values, error_values, 'r-', label='Error Rate')
                axs[2].set_ylabel('Error Rate')
                axs[2].set_xlabel('Time')
                axs[2].grid(True)
                axs[2].legend()
            
            # Add annotation for the suspected issue time
            if len(x_values) > 1:
                issue_point = int(len(x_values) * 0.7)  # Assume issue at 70% of timeline
                for ax in axs:
                    ax.axvline(x=issue_point, color='r', linestyle='--', alpha=0.5)
                    ax.text(issue_point, ax.get_ylim()[1] * 0.9, 'Issue Detected', 
                           rotation=90, verticalalignment='top')
            
            # Add title with root cause information
            plt.suptitle(f"Issue Analysis for {issue_id}\n"
                       f"Root Cause: {root_cause.get('description', 'Unknown')}")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the figure
            file_path = os.path.join(vis_dir, f"{issue_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(file_path)
            plt.close(fig)
            
            logger.info(f"Created visualization at {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return "visualization_failed"
    
    def _save_analysis(self, analysis: Dict[str, Any], issue_id: str) -> None:
        """
        Save the analysis results to a file.
        
        Args:
            analysis: Analysis results dictionary
            issue_id: Issue ID
        """
        try:
            # Create analyses directory if it doesn't exist
            analyses_dir = os.path.join('data', 'analyses')
            os.makedirs(analyses_dir, exist_ok=True)
            
            # Save analysis to JSON file
            filename = f"{issue_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(analyses_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}") 