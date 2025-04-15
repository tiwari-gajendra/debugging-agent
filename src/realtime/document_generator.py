"""
Document Generator for BIM reports
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import time
import logging

logger = logging.getLogger(__name__)

class DocumentGenerator:
    def __init__(self):
        self.template_path = Path("data/templates/bim_template.md")
        self.reports_path = Path("data/reports")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
    def _load_template(self) -> str:
        """Load the BIM template."""
        with open(self.template_path, 'r') as f:
            return f.read()
            
    def _load_logs(self, service_name: str) -> List[Dict[str, Any]]:
        """Load logs for the given service."""
        log_path = Path(f"data/logs/service_logs/{service_name}_service.json")
        with open(log_path, 'r') as f:
            return json.load(f)
            
    def _analyze_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze logs and extract relevant information."""
        # Get the first log entry with warnings
        log_entry = None
        warnings = []
        errors = []
        
        for entry in logs:
            log_data = json.loads(json.loads(entry['line'])['log'])
            
            # Store the first entry for service info
            if not log_entry:
                log_entry = entry
                
            # Collect warnings
            if 'webhookpost Response' in log_data and log_data['webhookpost Response'].get('warnings'):
                warnings.extend(log_data['webhookpost Response']['warnings'])
                
            # Collect errors
            if log_data.get('level') == 'error':
                errors.append(log_data)
                
        if not log_entry:
            raise ValueError("No valid log entries found")
            
        # Get service info from the first entry
        service_info = {
            'service': log_entry['fields']['app_kubernetes_io_instance'],
            'cluster': log_entry['fields']['cluster'],
            'pod': log_entry['fields']['pod'],
            'container': log_entry['fields']['container']
        }
        
        # Get metrics from the first entry with webhookpost Response
        for entry in logs:
            log_data = json.loads(json.loads(entry['line'])['log'])
            if 'webhookpost Response' in log_data:
                webhook_response = log_data['webhookpost Response']
                service_info.update({
                    'daily_bid_requests': webhook_response['daily_bid_requests'],
                    'total_bid_requests': webhook_response['total_bid_requests'],
                    'unique_devices': webhook_response['unique_devices'],
                    'unique_ips': webhook_response['unique_ips']
                })
                break
                
        service_info['warnings'] = warnings
        service_info['errors'] = errors
        return service_info
        
    def _format_warnings(self, warnings: list) -> str:
        """Format warning messages."""
        formatted = "**Warning Messages Found:**\n\n"
        for warning in warnings:
            formatted += f"- Code: {', '.join(warning['code'])}\n"
            formatted += f"  Message: {', '.join(warning['message'])}\n"
        return formatted
        
    def _format_errors(self, errors: list) -> str:
        """Format error messages."""
        if not errors:
            return "No errors found in the logs."
            
        formatted = "**Error Messages Found:**\n\n"
        for error in errors:
            formatted += f"- Level: {error['level']}\n"
            formatted += f"  Message: {error['msg']}\n"
            if 'error' in error:
                formatted += f"  Details: {error['error']}\n"
        return formatted
        
    def _convert_format(self, input_file: Path, output_format: str) -> Path:
        """Convert document to desired format."""
        try:
            import pypandoc
            output_file = input_file.parent / f"{input_file.stem}.{output_format}"
            
            # Convert using pandoc
            pypandoc.convert_file(
                str(input_file),
                output_format,
                outputfile=str(output_file),
                format=input_file.suffix[1:]  # Remove the dot from suffix
            )
            
            return output_file
        except ImportError:
            logger.error("pypandoc not installed. Please install it for format conversion.")
            raise
        except Exception as e:
            logger.error(f"Error converting document format: {str(e)}")
            raise
    
    def generate_bim_doc(self, ticket_data: dict, output_format: str = 'doc') -> str:
        """
        Generate a BIM document from ticket data.
        
        Args:
            ticket_data: Dictionary containing ticket data or template
            output_format: Output format (doc, pdf, markdown)
            
        Returns:
            Path to generated document
        """
        try:
            # Check if we're using a template
            if 'template' in ticket_data:
                content = ticket_data['template']
            else:
                # Try to use ticket data
                template_path = Path("data/templates/bim_template.md")
                if not template_path.exists():
                    raise FileNotFoundError("BIM template not found")
                
                with open(template_path, 'r') as f:
                    template = f.read()
                
                # Replace placeholders with ticket data
                content = template.format(
                    ticket_id=ticket_data.get('key', 'UNKNOWN'),
                    summary=ticket_data.get('summary', 'No summary available'),
                    description=ticket_data.get('description', 'No description available'),
                    status=ticket_data.get('status', 'Unknown'),
                    priority=ticket_data.get('priority', 'Unknown'),
                    created=ticket_data.get('created', 'Unknown'),
                    updated=ticket_data.get('updated', 'Unknown')
                )
            
            # Create output directory if it doesn't exist
            output_dir = Path("data/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            timestamp = int(time.time())
            ticket_id = ticket_data.get('key', 'UNKNOWN')
            
            # First save as markdown
            md_file = output_dir / f"BIM_{ticket_id}_{timestamp}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Convert to desired format if different from markdown
            if output_format != 'md':
                output_file = self._convert_format(md_file, output_format)
                return str(output_file)
            
            return str(md_file)
            
        except Exception as e:
            logger.error(f"Error generating BIM document: {str(e)}")
            raise RuntimeError(f"Failed to generate document: {str(e)}")
    
    def get_task_description(self, issue_id: str) -> str:
        """
        Get a description of the task for this agent.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Task description string
        """
        return (f"Generate comprehensive documentation for issue {issue_id}, including analysis results, "
                f"debugging steps taken, and recommendations.") 