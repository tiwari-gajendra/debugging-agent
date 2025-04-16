"""
JIRA Client Integration

This module provides functionality to fetch JIRA ticket data either from the JIRA API
or from a fallback Word document if the API is unavailable.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import requests
from dotenv import load_dotenv
import aiohttp

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class JiraClient:
    """Client for interacting with JIRA API with local fallback."""
    
    def __init__(self):
        """Initialize the JIRA client."""
        self.jira_url = os.getenv("JIRA_URL")
        self.jira_token = os.getenv("JIRA_TOKEN")
        self.templates_dir = Path("data/test_data/jira")
        self.base_url = os.getenv('JIRA_BASE_URL')
        self.auth = (
            os.getenv('JIRA_EMAIL', ''),
            os.getenv('JIRA_API_TOKEN', '')
        )
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)
        self.fallback_dir = Path('data/test_data')
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
    def create_issue(self, project_key: str, summary: str, description: str, 
                    issue_type: str = "Bug", priority: str = "High",
                    labels: List[str] = None) -> Dict:
        """Create a new JIRA issue."""
        try:
            url = f"{self.base_url}/rest/api/3/issue"
            
            payload = {
                "fields": {
                    "project": {"key": project_key},
                    "summary": summary,
                    "description": description,
                    "issuetype": {"name": issue_type},
                    "priority": {"name": priority}
                }
            }
            
            if labels:
                payload["fields"]["labels"] = labels
                
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                auth=self.auth
            )
            
            if response.status_code == 201:
                self.logger.info(f"Successfully created JIRA issue: {response.json()['key']}")
                return response.json()
            else:
                self.logger.error(f"Failed to create JIRA issue: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating JIRA issue: {str(e)}")
            return None
            
    def add_comment(self, issue_key: str, comment: str) -> bool:
        """Add a comment to an existing JIRA issue."""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{issue_key}/comment"
            
            payload = {
                "body": comment
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                auth=self.auth
            )
            
            if response.status_code == 201:
                self.logger.info(f"Successfully added comment to issue {issue_key}")
                return True
            else:
                self.logger.error(f"Failed to add comment to issue {issue_key}: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding comment to JIRA issue: {str(e)}")
            return False
            
    def update_issue_status(self, issue_key: str, status: str) -> bool:
        """Update the status of a JIRA issue."""
        try:
            # First, get the transition ID for the desired status
            url = f"{self.base_url}/rest/api/3/issue/{issue_key}/transitions"
            response = requests.get(url, headers=self.headers, auth=self.auth)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get transitions for issue {issue_key}: {response.text}")
                return False
                
            transitions = response.json()["transitions"]
            transition_id = None
            
            for transition in transitions:
                if transition["to"]["name"].lower() == status.lower():
                    transition_id = transition["id"]
                    break
                    
            if not transition_id:
                self.logger.error(f"Could not find transition to status: {status}")
                return False
                
            # Perform the transition
            url = f"{self.base_url}/rest/api/3/issue/{issue_key}/transitions"
            payload = {
                "transition": {"id": transition_id}
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                auth=self.auth
            )
            
            if response.status_code == 204:
                self.logger.info(f"Successfully updated issue {issue_key} status to {status}")
                return True
            else:
                self.logger.error(f"Failed to update issue {issue_key} status: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating JIRA issue status: {str(e)}")
            return False
            
    def get_issue(self, issue_key: str) -> Optional[Dict]:
        """Get details of a JIRA issue."""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
            response = requests.get(url, headers=self.headers, auth=self.auth)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get issue {issue_key}: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting JIRA issue: {str(e)}")
            return None
        
    async def fetch_ticket(self, ticket_id: str) -> Dict[str, Any]:
        """
        Fetch JIRA ticket information with fallback to local templates.
        
        Args:
            ticket_id: The JIRA ticket ID
            
        Returns:
            Dictionary containing ticket information
        """
        # Try JIRA API first
        if self.jira_url and self.jira_token:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {self.jira_token}",
                        "Content-Type": "application/json"
                    }
                    async with session.get(
                        f"{self.jira_url}/rest/api/2/issue/{ticket_id}",
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        logger.warning(f"JIRA API returned status {response.status}")
            except Exception as e:
                logger.warning(f"Error accessing JIRA API: {e}")
        
        # Fallback to local template
        template_path = self.templates_dir / f"{ticket_id}.doc"
        if template_path.exists():
            try:
                return self._parse_doc_file(template_path, ticket_id)
            except Exception as e:
                logger.error(f"Error reading local template: {e}")
                return {"error": f"Error reading local template: {str(e)}"}
        
        # Return graceful error if no data available
        return {
            "error": (
                f"Could not fetch ticket {ticket_id}. "
                "JIRA API is not accessible and no local template found."
            )
        }
        
    def _parse_doc_file(self, file_path: Path, ticket_id: str) -> Dict[str, Any]:
        """
        Parse a .doc file containing ticket information into a structured dictionary.
        
        Args:
            file_path: Path to the .doc file
            ticket_id: The JIRA ticket ID
            
        Returns:
            Dictionary containing ticket information in a structured format
        """
        self.logger.info(f"Parsing DOC file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Extract summary (first line)
            summary = lines[0].strip() if lines else "No summary available"
            
            # Extract description (everything except the first line)
            description = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
            
            # Extract other fields from the description if possible
            priority = "High"  # Default
            status = "Open"    # Default
            components = []
            
            # Look for key points section to extract components
            if "Key points:" in description:
                key_points_section = description.split("Key points:")[1].split("\n\n")[0]
                components_found = []
                
                for line in key_points_section.split("\n"):
                    if line.strip().startswith("-"):
                        component_name = line.strip().replace("-", "").strip().split(" ")[0]
                        if component_name:
                            components_found.append({"name": component_name})
                
                if components_found:
                    components = components_found
            
            # Build a structured response that matches what would come from JIRA API
            return {
                "key": ticket_id,
                "fields": {
                    "summary": summary,
                    "description": description,
                    "issuetype": {"name": "Bug"},  # Default
                    "priority": {"name": priority},
                    "status": {"name": status},
                    "created": datetime.now().isoformat(),
                    "reporter": {"displayName": "Local Tester"},
                    "environment": "Test Environment",
                    "components": components if components else [{"name": "Unknown"}]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing DOC file: {e}")
            raise
        
    async def validate_ticket_exists(self, ticket_id: str) -> Dict[str, Any]:
        """
        Validate if a JIRA ticket exists either via API or in local test data.
        
        Args:
            ticket_id: The JIRA ticket ID to validate
            
        Returns:
            Dictionary with validation results containing:
            {
                "exists": bool,  # True if ticket can be accessed (via API or test data)
                "message": str,  # Information about validation result
                "source": str,   # "api" if using live JIRA, "local" if using test data, None if invalid
                "data": dict     # The ticket data if exists=True, otherwise None
            }
        """
        self.logger.info(f"Validating JIRA ticket: {ticket_id}")
        
        result = {
            "exists": False,
            "message": "",
            "source": None,
            "data": None
        }
        
        # First attempt to use the JIRA API
        if self.jira_url and self.jira_token:
            try:
                ticket_data = await self.fetch_ticket(ticket_id)
                
                # Check if we got an error response
                if "error" not in ticket_data:
                    self.logger.info(f"Successfully validated ticket {ticket_id} via JIRA API")
                    result["exists"] = True
                    result["message"] = f"Using JIRA API for ticket {ticket_id}"
                    result["source"] = "api"
                    result["data"] = ticket_data
                    return result
            except Exception as e:
                self.logger.warning(f"Could not validate ticket via JIRA API: {str(e)}")
        
        # Then check for local test data
        template_path = self.templates_dir / f"{ticket_id}.doc"
        if template_path.exists():
            try:
                ticket_data = self._parse_doc_file(template_path, ticket_id)
                self.logger.info(f"Found local test data for ticket {ticket_id}")
                result["exists"] = True
                result["message"] = f"Using local test data for ticket {ticket_id}"
                result["source"] = "local"
                result["data"] = ticket_data
                return result
            except Exception as e:
                self.logger.error(f"Error reading local template: {e}")
                result["message"] = f"Error reading local template: {str(e)}"
                return result
        
        # If we get here, the ticket doesn't exist in API or local test data
        self.logger.warning(f"Could not validate ticket {ticket_id} - not found in API or test data")
        result["message"] = f"Ticket {ticket_id} not found. Please verify the ticket ID."
        return result
        
    def _fetch_from_local(self, ticket_id: str) -> Optional[Dict]:
        """Fetch ticket data from local storage."""
        try:
            file_path = self.fallback_dir / f"{ticket_id}.doc"
            if file_path.exists():
                return self._parse_doc_file(file_path, ticket_id)
        except Exception as e:
            self.logger.warning(f"Error reading local ticket data: {e}")
        return None
        
    async def _fetch_from_api(self, ticket_id: str) -> Optional[Dict]:
        """Fetch ticket data from JIRA API."""
        try:
            url = f"{self.base_url}/rest/api/3/issue/{ticket_id}"
            response = requests.get(url, headers=self.headers, auth=self.auth)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to fetch JIRA ticket: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching JIRA ticket: {str(e)}")
            return None
            
    def _get_dummy_data(self, ticket_id: str) -> Dict:
        """Get dummy ticket data for testing."""
        return {
            "key": ticket_id,
            "fields": {
                "summary": "Forecasting Engine Warning Messages in Production",
                "description": "The forecasting engine is generating warning messages in production for date range validations.",
                "issuetype": {"name": "Bug"},
                "priority": {"name": "High"},
                "status": {"name": "Open"},
                "created": datetime.now().isoformat(),
                "reporter": {"displayName": "John Doe"},
                "environment": "Production",
                "components": [{"name": "Forecasting Engine"}]
            }
        } 