"""
Debug Crew Manager - Manages crew of debugging agents.
"""

import os
import sys
import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

from src.utils.llm_factory import LLMFactory

# Load environment variables
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)

class DebugCrew:
    """Manages a crew of specialized debugging agents."""
    
    def __init__(self, llm_provider_or_model: Optional[str] = None):
        """
        Initialize the Debug Crew.
        
        Args:
            llm_provider_or_model: The LLM provider or model name to use (defaults to env var LLM_PROVIDER)
        """
        # Get the provider and create LLM instance using factory
        self.provider = (llm_provider_or_model or os.getenv('LLM_PROVIDER', 'openai')).split('#')[0].strip().lower()
        logger.info(f"Initializing DebugCrew with LLM provider: {self.provider}")
        
        # Create LLM from factory
        self.llm = LLMFactory.create_llm(provider=self.provider)
        
        # Initialize empty agent and task lists
        self.agents = []
        self.tasks = []
    
    def add_agents(self, agents: List[Any]) -> None:
        """
        Add agents to the crew.
        
        Args:
            agents: List of agent objects to add to the crew
        """
        for agent_obj in agents:
            # Convert each agent object to a crewAI Agent
            agent_name = agent_obj.__class__.__name__
            agent_role = agent_name.replace('Builder', '').replace('Creator', '')
            
            logger.debug(f"Creating CrewAI agent for {agent_name}")
            
            # Create agent config based on provider type
            agent_config = {
                "name": agent_name,
                "role": f"{agent_role} Specialist",
                "goal": f"Provide expert {agent_role.lower()} support for debugging issues",
                "backstory": f"You are an expert in {agent_role.lower()} for software systems, "
                           f"with years of experience diagnosing and fixing complex issues.",
                "llm": self.llm
            }
            
            # Create the agent with the appropriate config
            from crewai import Agent
            agent = Agent(**agent_config)
            
            # Store both the crewAI agent and the original agent object
            self.agents.append({
                "crew_agent": agent,
                "agent_obj": agent_obj,
                "agent_name": agent_name  # Store name explicitly for logging
            })
            
            logger.debug(f"Added agent {agent_name} to crew")
    
    async def run(self, issue_id: str) -> Dict[str, Any]:
        """
        Run the debugging process.
        
        Args:
            issue_id: The issue ID to debug
            
        Returns:
            Dict with results information
        """
        logger.info(f"Running debugging process for issue {issue_id}")
        
        try:
            # Import Task here to avoid circular imports
            from crewai import Task, Crew, Process
            
            # Create tasks for each agent
            for agent_info in self.agents:
                agent_obj = agent_info["agent_obj"]
                agent_name = agent_info["agent_name"]
                
                # Get task description from the agent object
                task_description = agent_obj.get_task_description(issue_id)
                
                # Create the task with expected output
                task = Task(
                    description=task_description,
                    agent=agent_info["crew_agent"],
                    expected_output=f"Analysis results for issue {issue_id} from {agent_name}"
                )
                
                self.tasks.append(task)
                logger.debug(f"Created task for agent {agent_name}")
            
            # Create and run the crew synchronously
            crew = Crew(
                agents=[a["crew_agent"] for a in self.agents],
                tasks=self.tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Run crew synchronously - no await needed
            result = crew.kickoff()
            
            # Get the path to the project root (two levels up from this file)
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(src_dir)
            
            # Ensure reports directory exists
            reports_dir = os.path.join(project_root, "data", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Create report filename
            timestamp = os.environ.get("TEST_TIMESTAMP") or int(datetime.now().timestamp())
            report_filename = f"debug_report_{issue_id}_{timestamp}.html"
            report_path = os.path.join(reports_dir, report_filename)
            
            logger.info(f"Generating report at {report_path}")
            
            # Write a basic HTML report with the crew output
            with open(report_path, "w") as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Debug Report: {issue_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .section {{ margin-bottom: 20px; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Debug Report: {issue_id}</h1>
    <div class="section">
        <h2>Debugging Results</h2>
        <pre>{str(result)}</pre>
    </div>
</body>
</html>""")
            
            logger.info(f"Report generated at {report_path}")
            
            # Process the result
            return {
                "crew_output": str(result),
                "document_url": report_path
            }
            
        except Exception as e:
            logger.error(f"Error running debugging process: {str(e)}")
            raise
