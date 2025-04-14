"""
Slack Handler - Processes Slack alerts and triggers RCA.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

import slack_sdk
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest

# Import debugging components
from src.manager.crew_manager import DebugCrew
from src.realtime.context_builder import ContextBuilder
from src.realtime.debug_plan_creator import DebugPlanCreator
from src.realtime.executor import Executor
from src.realtime.analyzer import Analyzer
from src.realtime.document_generator import DocumentGenerator

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class SlackHandler:
    """Handles Slack alerts and triggers RCA."""
    
    def __init__(self):
        """Initialize the Slack handler."""
        # Get Slack credentials
        self.slack_app_token = os.getenv('SLACK_APP_TOKEN')
        self.slack_bot_token = os.getenv('SLACK_BOT_TOKEN')
        
        if not self.slack_app_token or not self.slack_bot_token:
            raise ValueError("Missing Slack credentials. Set SLACK_APP_TOKEN and SLACK_BOT_TOKEN in .env")
        
        # Initialize Slack client
        self.web_client = slack_sdk.WebClient(token=self.slack_bot_token)
        self.socket_client = SocketModeClient(
            app_token=self.slack_app_token,
            web_client=self.web_client
        )
        
        # Initialize debugging components
        self.debug_crew = DebugCrew()
        self.context_builder = ContextBuilder()
        self.plan_creator = DebugPlanCreator()
        self.executor = Executor()
        self.analyzer = Analyzer()
        self.document_generator = DocumentGenerator()
        
        # Add agents to crew
        self.debug_crew.add_agents([
            self.context_builder,
            self.plan_creator,
            self.executor,
            self.analyzer,
            self.document_generator
        ])
        
        logger.info("Initialized SlackHandler")
    
    def start(self):
        """Start listening for Slack events."""
        self.socket_client.socket_mode_request_listeners.append(self.handle_socket_request)
        self.socket_client.connect()
        logger.info("Started listening for Slack events")
    
    async def handle_socket_request(self, client: SocketModeClient, req: SocketModeRequest):
        """
        Handle incoming Slack socket mode requests.
        
        Args:
            client: Socket mode client
            req: Socket mode request
        """
        # Acknowledge the request
        response = SocketModeResponse(envelope_id=req.envelope_id)
        client.send_socket_mode_response(response)
        
        # Process the event
        if req.type == "events_api":
            event = req.payload.get("event", {})
            if event.get("type") == "message" and event.get("subtype") != "bot_message":
                await self.handle_message(event)
    
    async def handle_message(self, event: Dict[str, Any]):
        """
        Handle incoming Slack messages.
        
        Args:
            event: Slack event data
        """
        text = event.get("text", "").lower()
        channel = event.get("channel")
        
        # Check if message contains alert keywords
        alert_keywords = ["error", "alert", "issue", "down", "failed", "critical"]
        if any(keyword in text for keyword in alert_keywords):
            logger.info(f"Detected alert in message: {text}")
            
            # Generate issue ID
            issue_id = f"SLACK-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Notify channel that RCA is starting
            await self.web_client.chat_postMessage(
                channel=channel,
                text=f"🚨 Alert detected! Starting RCA for issue {issue_id}..."
            )
            
            try:
                # Run RCA
                result = await self.run_rca(issue_id, text)
                
                # Post results
                if result.get("success"):
                    report_path = result.get("report_path")
                    await self.web_client.chat_postMessage(
                        channel=channel,
                        text=f"✅ RCA completed for issue {issue_id}.\nReport: {report_path}"
                    )
                else:
                    await self.web_client.chat_postMessage(
                        channel=channel,
                        text=f"❌ RCA failed for issue {issue_id}: {result.get('error')}"
                    )
                    
            except Exception as e:
                logger.error(f"Error running RCA: {e}")
                await self.web_client.chat_postMessage(
                    channel=channel,
                    text=f"❌ Error running RCA: {str(e)}"
                )
    
    async def run_rca(self, issue_id: str, alert_text: str) -> Dict[str, Any]:
        """
        Run Root Cause Analysis for an alert.
        
        Args:
            issue_id: Issue ID
            alert_text: Alert message text
            
        Returns:
            Dictionary with RCA results
        """
        try:
            # Build context
            context = await self.context_builder.build_context(issue_id)
            
            # Add alert text to context
            context["alert_text"] = alert_text
            
            # Run debugging process
            result = await self.debug_crew.run(issue_id=issue_id)
            
            return {
                "success": True,
                "report_path": result.get("report_path", "unknown"),
                "issue_id": issue_id
            }
            
        except Exception as e:
            logger.error(f"Error in RCA process: {e}")
            return {
                "success": False,
                "error": str(e),
                "issue_id": issue_id
            } 