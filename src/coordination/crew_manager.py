"""
Debug Crew Manager - Manages crew of debugging agents.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from datetime import datetime

from src.utils.llm_factory import LLMFactory, DirectBedrockClient

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
        self.provider_or_model = llm_provider_or_model or os.getenv('LLM_PROVIDER', 'openai')
        logger.info(f"Initializing DebugCrew with LLM provider/model: {self.provider_or_model}")
        
        # Create an LLM instance
        self.llm = LLMFactory.create_llm(self.provider_or_model)
        
        # Determine if we're using Bedrock provider
        is_bedrock = False
        if self.provider_or_model == 'bedrock':
            is_bedrock = True
        else:
            # Check if we're using a model from the registry that's on Bedrock
            models = LLMFactory.list_available_models()
            if self.provider_or_model.lower() in models and models[self.provider_or_model.lower()] == 'bedrock':
                is_bedrock = True
        
        # Note: We don't need to set environment variables here anymore
        # They are handled in the DirectBedrockClient class
        
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
                           f"with years of experience diagnosing and fixing complex issues."
            }
            
            # Determine if we're using Bedrock provider
            is_bedrock = False
            if self.provider_or_model == 'bedrock':
                is_bedrock = True
            else:
                # Check if we're using a model from the registry that's on Bedrock
                models = LLMFactory.list_available_models()
                if self.provider_or_model.lower() in models and models[self.provider_or_model.lower()] == 'bedrock':
                    is_bedrock = True
            
            # For Bedrock, we need to use the model parameter instead of llm
            if is_bedrock and isinstance(self.llm, DirectBedrockClient):
                # Use a different agent creation method for Bedrock
                logger.info(f"Creating Bedrock-compatible agent for {agent_name}")
                agent_config["model"] = self.llm
                agent_config["llm_provider"] = "bedrock"
                
                # Register our API key for Bedrock
                os.environ["OPENAI_API_KEY"] = "sk-valid-key"
            else:
                # For all other providers, use the normal llm parameter
                agent_config["llm"] = self.llm
            
            # Create the agent with the appropriate config
            agent = Agent(**agent_config)
            
            # Store both the crewAI agent and the original agent object
            self.agents.append({
                "crew_agent": agent,
                "agent_obj": agent_obj,
                "agent_name": agent_name  # Store name explicitly for logging
            })
            
            logger.debug(f"Added agent {agent_name} to crew")
    
    def run(self, issue_id: str) -> Dict[str, Any]:
        """
        Run the debugging process for an issue.
        
        Args:
            issue_id: The ID of the issue to debug
            
        Returns:
            Dict containing results of the debugging process
        """
        if not self.agents:
            raise ValueError("No agents added to the crew")
        
        logger.info(f"Running debugging process for issue {issue_id}")
        
        # Create tasks for each agent
        tasks = []
        
        # Create properly structured context
        context = [{
            "issue_id": issue_id,
            "description": f"Debug information for issue {issue_id}",
            "expected_output": f"Analysis results for issue {issue_id}"
        }]
        
        # Create tasks based on the agent sequence
        for i, agent_pair in enumerate(self.agents):
            agent = agent_pair["crew_agent"]
            agent_obj = agent_pair["agent_obj"]
            agent_name = agent_pair["agent_name"]  # Get stored name
            
            # Create a task for this agent
            task_description = f"Analyze and process the issue {issue_id}"
            if hasattr(agent_obj, "get_task_description"):
                task_description = agent_obj.get_task_description(issue_id)
            
            task = Task(
                description=task_description,
                agent=agent,
                context=context,
                expected_output=f"Analysis results for issue {issue_id}"
            )
            tasks.append(task)
            
            # For debugging later, store tasks
            self.tasks.append(task)
            
            logger.debug(f"Created task for agent {agent_name}")
        
        # Create crew-specific config based on provider
        crew_config = {
            "agents": [a["crew_agent"] for a in self.agents],
            "tasks": tasks,
            "process": Process.sequential,
            "verbose": True
        }
        
        # Determine if we're using Bedrock provider
        is_bedrock = False
        if self.provider_or_model == 'bedrock':
            is_bedrock = True
        else:
            # Check if we're using a model from the registry that's on Bedrock
            models = LLMFactory.list_available_models()
            if self.provider_or_model.lower() in models and models[self.provider_or_model.lower()] == 'bedrock':
                is_bedrock = True
        
        # Add provider override for Bedrock
        if is_bedrock:
            crew_config["llm_provider"] = "bedrock"
            
        # Create the crew with sequential process
        crew = Crew(**crew_config)
        
        logger.info("Starting crew kickoff")
        
        try:
            # Run the crew
            crew_output = crew.kickoff()
            logger.info("Crew execution completed successfully")
        except Exception as e:
            logger.error(f"Error during crew execution: {str(e)}")
            crew_output = f"Error during execution: {str(e)}"
        
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
        <pre>{str(crew_output)}</pre>
    </div>
</body>
</html>""")
        
        # Create a dictionary with the crew output and document URL
        result = {
            "crew_output": crew_output,
            "document_url": f"file://{report_path}"
        }
        
        logger.info(f"Report generated at {report_path}")
        
        return result 