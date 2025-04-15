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
        template_path = self.templates_dir / f"{ticket_id}.json"
        if template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    return json.load(f)
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
        
    def _fetch_from_local(self, ticket_id: str) -> Optional[Dict]:
        """Fetch ticket data from local storage."""
        try:
            file_path = self.fallback_dir / f"{ticket_id}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
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