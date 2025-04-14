"""
Document Generator - Generates debugging reports.
"""

import os
import json
import logging
import textwrap
from typing import Dict, List, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

# Import template manager
from src.utils.template_manager import TemplateManager

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class DocumentGenerator:
    """
    Agent that generates comprehensive debugging reports based on analysis results.
    """
    
    def __init__(self, output_format: str = "md", template_name: Optional[str] = None):
        """
        Initialize the DocumentGenerator agent.
        
        Args:
            output_format: Format for generated documents (md, markdown)
            template_name: Name of the template to use (defaults to rca_template.json)
        """
        self.output_format = output_format.lower()
        if self.output_format not in ["md", "markdown"]:
            raise ValueError("Only markdown format is supported")
            
        self.template_name = template_name or "rca_template.json"
        self.template_manager = TemplateManager()
        
        logger.info(f"Initializing DocumentGenerator with format: {output_format}, template: {self.template_name}")
        
        # Get the path to the project root directory (3 levels up from this file)
        self.src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.dirname(self.src_dir)
    
    def get_task_description(self, issue_id: str) -> str:
        """
        Get a description of the task for this agent.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Task description string
        """
        return textwrap.dedent(f"""
        Generate a comprehensive debugging report for issue {issue_id} based on:
        1. The context information gathered
        2. The debugging plan created
        3. The execution results collected
        4. The analysis performed
        
        The report should include an executive summary, detailed analysis, and recommendations.
        """).strip()
    
    def generate_report(self,
                       issue_id: str,
                       context: Dict[str, Any],
                       plan: Dict[str, Any],
                       execution_result: Dict[str, Any],
                       analysis: Dict[str, Any]) -> str:
        """
        Generate a debugging report.
        
        Args:
            issue_id: Issue ID
            context: Context information
            plan: Debugging plan
            execution_result: Execution results
            analysis: Analysis results
            
        Returns:
            Path to the generated report file
        """
        try:
            # Load template
            template = self.template_manager.load_template(self.template_name)
            
            # Prepare data for template
            data = {
                "issue_id": issue_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "alert_text": context.get("alert_text", "N/A"),
                "root_cause": analysis.get("root_cause", {}).get("description", "Unknown"),
                "confidence": analysis.get("confidence", "low"),
                "immediate_actions": "\n".join(f"- {action}" for action in analysis.get("solution", {}).get("immediate_actions", [])),
                "long_term_actions": "\n".join(f"- {action}" for action in analysis.get("solution", {}).get("long_term_actions", [])),
                "error_logs": "\n".join(f"- {log['message']}" for log in context.get("logs", []) if log.get("level") in ["ERROR", "WARN"]),
                "metrics_summary": self._format_metrics_summary(context.get("metrics", {})),
                "trace_summary": self._format_trace_summary(context.get("traces", []))
            }
            
            # Apply template
            report_content = self.template_manager.apply_template(template, data)
            
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(self.project_root, 'data', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{issue_id}_report_{timestamp}.{self.output_format}"
            filepath = os.path.join(reports_dir, filename)
            
            # Save report
            with open(filepath, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Generated report at {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _format_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics data for the report.
        
        Args:
            metrics: Metrics data
            
        Returns:
            Formatted metrics summary
        """
        summary = []
        
        # CPU usage
        cpu_values = metrics.get("cpu_usage", [])
        if cpu_values:
            avg_cpu = sum(cpu_values) / len(cpu_values)
            max_cpu = max(cpu_values)
            summary.append(f"CPU Usage: Average {avg_cpu:.1f}%, Max {max_cpu:.1f}%")
        
        # Memory usage
        memory_values = metrics.get("memory_usage", [])
        if memory_values:
            avg_mem = sum(memory_values) / len(memory_values)
            max_mem = max(memory_values)
            summary.append(f"Memory Usage: Average {avg_mem:.1f}%, Max {max_mem:.1f}%")
        
        # Error rate
        error_values = metrics.get("error_rate", [])
        if error_values:
            avg_error = sum(error_values) / len(error_values)
            max_error = max(error_values)
            summary.append(f"Error Rate: Average {avg_error:.2f}, Max {max_error:.2f}")
        
        return "\n".join(summary)
    
    def _format_trace_summary(self, traces: List[Dict[str, Any]]) -> str:
        """
        Format trace data for the report.
        
        Args:
            traces: Trace data
            
        Returns:
            Formatted trace summary
        """
        summary = []
        
        # Count failed traces
        failed_traces = [trace for trace in traces 
                        if any(span.get("status") == "error" for span in trace.get("spans", []))]
        
        if failed_traces:
            summary.append(f"Found {len(failed_traces)} failed traces")
            
            # Group by service
            services = {}
            for trace in failed_traces:
                for span in trace.get("spans", []):
                    if span.get("status") == "error":
                        service = span.get("service", "unknown")
                        if service not in services:
                            services[service] = 0
                        services[service] += 1
            
            # Add service breakdown
            for service, count in services.items():
                summary.append(f"- {service}: {count} errors")
        
        return "\n".join(summary) if summary else "No trace issues found" 