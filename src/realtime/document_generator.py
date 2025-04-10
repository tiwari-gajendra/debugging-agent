"""
Document Generator - Generates debugging reports.
"""

import os
import json
import logging
import textwrap
from typing import Dict, List, Any, Optional
from datetime import datetime
import html

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class DocumentGenerator:
    """
    Agent that generates comprehensive debugging reports based on analysis results.
    """
    
    def __init__(self, output_format: str = "html"):
        """
        Initialize the DocumentGenerator agent.
        
        Args:
            output_format: Format for generated documents (html, markdown, etc.)
        """
        self.output_format = output_format.lower()
        logger.info(f"Initializing DocumentGenerator with format: {output_format}")
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
    
    def generate_document(self, 
                         context: Dict[str, Any],
                         plan: Dict[str, Any],
                         execution_result: Dict[str, Any],
                         analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive debugging report.
        
        Args:
            context: Context information from ContextBuilder
            plan: Debugging plan from DebugPlanCreator
            execution_result: Results from Executor
            analysis: Analysis results from Analyzer
            
        Returns:
            Dictionary with document details and path
        """
        issue_id = context.get("issue_id", "unknown")
        logger.info(f"Generating debugging report for issue {issue_id}")
        
        if self.output_format == "html":
            document, path = self._generate_html_report(
                issue_id, context, plan, execution_result, analysis
            )
        else:
            document, path = self._generate_html_report(  # Default to HTML
                issue_id, context, plan, execution_result, analysis
            )
        
        # Create document details
        document_details = {
            "issue_id": issue_id,
            "document_id": f"doc-{issue_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "format": self.output_format,
            "path": path,
            "url": f"file://{os.path.abspath(path)}",  # Local file URL
            "document": document if self.output_format != "html" else None  # Don't include HTML in JSON
        }
        
        logger.info(f"Generated debugging report saved to {path}")
        return document_details
    
    def _generate_html_report(self,
                             issue_id: str,
                             context: Dict[str, Any],
                             plan: Dict[str, Any],
                             execution_result: Dict[str, Any],
                             analysis: Dict[str, Any]) -> tuple:
        """
        Generate an HTML debugging report.
        
        Args:
            issue_id: Issue ID
            context: Context information
            plan: Debugging plan
            execution_result: Execution results
            analysis: Analysis results
            
        Returns:
            Tuple of (HTML content, file path)
        """
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(self.project_root, 'data', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{issue_id}_report_{timestamp}.html"
        filepath = os.path.join(reports_dir, filename)
        
        # Generate report time info
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get root cause and solution
        root_cause = analysis.get("root_cause", {})
        root_cause_desc = root_cause.get("description", "Unknown")
        root_cause_confidence = analysis.get("confidence", "low")
        solution = analysis.get("solution", {})
        
        # Get chart path
        chart_path = analysis.get("visualization_path", "")
        chart_rel_path = os.path.relpath(chart_path, reports_dir) if chart_path and chart_path != "visualization_failed" else ""
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Debug Report: {issue_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #34495e;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .section {{
                    background-color: #f9f9f9;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    border-left: 5px solid #3498db;
                }}
                .summary {{
                    background-color: #e8f4fc;
                    border-left: 5px solid #2980b9;
                }}
                .analysis {{
                    background-color: #f7f9fe;
                    border-left: 5px solid #9b59b6;
                }}
                .solution {{
                    background-color: #e8f8f5;
                    border-left: 5px solid #27ae60;
                }}
                .execution {{
                    background-color: #fef9e7;
                    border-left: 5px solid #f1c40f;
                }}
                .high {{
                    color: #c0392b;
                    font-weight: bold;
                }}
                .medium {{
                    color: #d35400;
                    font-weight: bold;
                }}
                .low {{
                    color: #27ae60;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 15px;
                }}
                th, td {{
                    padding: 10px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .chart {{
                    max-width: 100%;
                    margin: 20px 0;
                }}
                pre {{
                    background-color: #f8f8f8;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Debugging Report for Issue: {issue_id}</h1>
                <p>Generated on: {report_time}</p>
            </div>
            
            <div class="section summary">
                <h2>Executive Summary</h2>
                <p><strong>Root Cause:</strong> <span class="{root_cause_confidence}">{root_cause_desc}</span> (Confidence: {root_cause_confidence})</p>
                <p><strong>Affected Service:</strong> {root_cause.get("service", "Unknown")}</p>
                <p><strong>Issue Type:</strong> {root_cause.get("type", "Unknown")}</p>
                
                <h3>Recommended Actions</h3>
                <h4>Immediate Actions:</h4>
                <ul>
        """
        
        # Add immediate actions
        for action in solution.get("immediate_actions", []):
            html_content += f"        <li>{html.escape(action)}</li>\n"
        
        html_content += """
                </ul>
                
                <h4>Long-term Actions:</h4>
                <ul>
        """
        
        # Add long-term actions
        for action in solution.get("long_term_actions", []):
            html_content += f"        <li>{html.escape(action)}</li>\n"
        
        html_content += """
                </ul>
            </div>
            
            <div class="section analysis">
                <h2>Analysis Results</h2>
        """
        
        # Add chart if available
        if chart_rel_path:
            html_content += f"""
                <h3>Visualization</h3>
                <img src="{chart_rel_path}" class="chart" alt="Analysis Visualization">
            """
        
        # Add error analysis
        error_analysis = analysis.get("error_analysis", {})
        html_content += f"""
                <h3>Error Analysis</h3>
                <p>Detected {error_analysis.get("error_count", 0)} errors/warnings</p>
                
                <h4>Services with Most Errors:</h4>
                <table>
                    <tr>
                        <th>Service</th>
                        <th>Error Count</th>
                    </tr>
        """
        
        for service, count in error_analysis.get("service_errors", []):
            html_content += f"""
                    <tr>
                        <td>{html.escape(service)}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h4>Common Error Patterns:</h4>
                <ul>
        """
        
        for pattern, count in error_analysis.get("common_patterns", []):
            html_content += f"""
                    <li>{html.escape(pattern)} <em>({count} occurrences)</em></li>
            """
        
        html_content += """
                </ul>
                
        """
        
        # Add performance analysis
        performance_analysis = analysis.get("performance_analysis", {})
        html_content += """
                <h3>Performance Analysis</h3>
                <h4>Service Latencies:</h4>
                <table>
                    <tr>
                        <th>Service</th>
                        <th>Average Duration (ms)</th>
                    </tr>
        """
        
        for service, duration in performance_analysis.get("service_latencies", []):
            html_content += f"""
                    <tr>
                        <td>{html.escape(service)}</td>
                        <td>{duration:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h4>Performance Outliers:</h4>
        """
        
        outliers = performance_analysis.get("outliers", [])
        if outliers:
            html_content += """
                <table>
                    <tr>
                        <th>Service</th>
                        <th>Average Duration (ms)</th>
                        <th>Overall Average (ms)</th>
                        <th>Ratio</th>
                    </tr>
            """
            
            for outlier in outliers:
                html_content += f"""
                        <tr>
                            <td>{html.escape(outlier.get("service", ""))}</td>
                            <td>{outlier.get("avg_duration", 0):.2f}</td>
                            <td>{outlier.get("overall_avg", 0):.2f}</td>
                            <td class="{'high' if outlier.get('ratio', 1) > 3 else 'medium'}">{outlier.get("ratio", 1):.2f}x</td>
                        </tr>
                """
                
            html_content += """
                </table>
            """
        else:
            html_content += """
                <p>No significant performance outliers detected</p>
            """
        
        # Add resource analysis
        resource_analysis = analysis.get("resource_analysis", {})
        
        cpu_stats = resource_analysis.get("cpu_stats", {})
        memory_stats = resource_analysis.get("memory_stats", {})
        has_cpu_issue = resource_analysis.get("has_cpu_issue", False)
        has_memory_issue = resource_analysis.get("has_memory_issue", False)
        
        cpu_class = "high" if has_cpu_issue else "low"
        mem_class = "high" if has_memory_issue else "low"
        
        html_content += f"""
                <h3>Resource Analysis</h3>
                <table>
                    <tr>
                        <th>Resource</th>
                        <th>Average</th>
                        <th>Maximum</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td>CPU Usage</td>
                        <td>{cpu_stats.get("avg", 0):.1f}%</td>
                        <td>{cpu_stats.get("max", 0):.1f}%</td>
                        <td class="{cpu_class}">{"High" if has_cpu_issue else "Normal"}</td>
                    </tr>
                    <tr>
                        <td>Memory Usage</td>
                        <td>{memory_stats.get("avg", 0):.1f}%</td>
                        <td>{memory_stats.get("max", 0):.1f}%</td>
                        <td class="{mem_class}">{"High" if has_memory_issue else "Normal"}</td>
                    </tr>
                </table>
                
                <p>CPU-Error Correlation: <strong>{"Yes" if resource_analysis.get("cpu_error_correlation", False) else "No"}</strong></p>
                <p>Memory-Error Correlation: <strong>{"Yes" if resource_analysis.get("memory_error_correlation", False) else "No"}</strong></p>
            </div>
            
            <div class="section execution">
                <h2>Execution Details</h2>
                <h3>Debugging Plan</h3>
                <p><strong>Priority:</strong> {plan.get("priority", "medium")}</p>
                <p><strong>Total Estimated Time:</strong> {plan.get("total_estimated_time_minutes", 0)} minutes</p>
                
                <h4>Steps Executed:</h4>
        """
        
        # Add execution steps
        for step in execution_result.get("results", []):
            step_id = step.get("step_id", "")
            step_title = step.get("title", "")
            
            html_content += f"""
                <div class="section">
                    <h5>Step {step_id}: {html.escape(step_title)}</h5>
                    <p><strong>Status:</strong> {step.get("status", "unknown")}</p>
                    
                    <h6>Actions and Findings:</h6>
                    <ul>
            """
            
            for action_result in step.get("action_results", []):
                action = action_result.get("action", "")
                html_content += f"""
                        <li>
                            <strong>{html.escape(action)}</strong>
                            <ul>
                """
                
                for finding in action_result.get("findings", []):
                    html_content += f"""
                                <li>{html.escape(finding)}</li>
                    """
                
                html_content += """
                            </ul>
                        </li>
                """
            
            html_content += """
                    </ul>
                </div>
            """
        
        # Add context summary
        html_content += """
            </div>
            
            <div class="section">
                <h2>Context Information</h2>
        """
        
        if "summary" in context:
            html_content += f"""
                <p>{html.escape(context.get("summary", ""))}</p>
            """
        
        # Add related issues
        related_issues = context.get("related_issues", [])
        if related_issues:
            html_content += """
                <h3>Related Issues:</h3>
                <table>
                    <tr>
                        <th>Issue ID</th>
                        <th>Title</th>
                        <th>Resolution</th>
                        <th>Similarity</th>
                    </tr>
            """
            
            for issue in related_issues:
                html_content += f"""
                        <tr>
                            <td>{html.escape(issue.get("issue_id", ""))}</td>
                            <td>{html.escape(issue.get("title", ""))}</td>
                            <td>{html.escape(issue.get("resolution", ""))}</td>
                            <td>{issue.get("similarity_score", 0):.2f}</td>
                        </tr>
                """
                
            html_content += """
                </table>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Document Information</h2>
                <p><strong>Generated:</strong> """ + report_time + """</p>
                <p><strong>Issue ID:</strong> """ + issue_id + """</p>
                <p><strong>File Path:</strong> """ + filepath + """</p>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return html_content, filepath 