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

from src.utils.llm_provider import LLMProvider

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
        # Clean provider string
        raw_provider = llm_provider_or_model or os.getenv('LLM_PROVIDER', 'openai')
        self.provider = raw_provider.split('#')[0].strip().lower()
        
        logger.info(f"Initializing DebugCrew with LLM provider: {self.provider}")
        
        # Create an LLM instance
        self.llm = LLMProvider.create_llm(provider=self.provider)
        
        # Special handling for Ollama to make it work with CrewAI
        if self.provider == 'ollama':
            # Use OpenAI provider with our API key, but configure it for Ollama
            os.environ["OPENAI_API_KEY"] = "sk-valid-ollama-key"
            os.environ["OPENAI_API_BASE"] = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            # Let CrewAI use OpenAI provider but with our Ollama base URL
            os.environ["CREW_LLM_PROVIDER"] = "openai"
        # Set up environment variables for CrewAI compatibility
        elif self.provider == 'bedrock' or self.provider == 'anthropic':
            os.environ["OPENAI_API_KEY"] = "sk-valid-bedrock-key"
            os.environ["CREW_LLM_PROVIDER"] = "bedrock"
        
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
            
            # Add provider specific configurations
            if self.provider == 'bedrock' or self.provider == 'anthropic':
                agent_config["llm_provider"] = "bedrock"
            elif self.provider == 'ollama':
                # Use OpenAI provider but with our Ollama URL
                agent_config["llm_provider"] = "openai"
            
            # Create the agent with the appropriate config
            agent = Agent(**agent_config)
            
            # Store both the crewAI agent and the original agent object
            self.agents.append({
                "crew_agent": agent,
                "agent_obj": agent_obj,
                "agent_name": agent_name  # Store name explicitly for logging
            })
            
            logger.debug(f"Added agent {agent_name} to crew")
    
    def execute_direct_ollama_task(self, task, context):
        """
        Execute a task directly with Ollama, bypassing CrewAI's LLM integration
        
        Args:
            task: The task object to execute
            context: The task context
            
        Returns:
            Task result as string
        """
        # Get the task description and context
        task_description = task.description
        
        # Get model name from environment or use default
        model = os.getenv('OLLAMA_MODEL', 'llama3')
        
        # Get agent information - CrewAI stores this differently in newer versions
        try:
            # Try to access directly if available
            agent_name = task.agent.name
            agent_role = task.agent.role
            agent_goal = task.agent.goal
            agent_backstory = task.agent.backstory
        except AttributeError:
            # If not directly available, use the agent object itself
            agent = task.agent
            agent_name = getattr(agent, 'name', 'Debugging Agent')
            agent_role = getattr(agent, 'role', 'Technical Specialist')
            agent_goal = getattr(agent, 'goal', 'Analyze and fix technical issues')
            agent_backstory = getattr(agent, 'backstory', 'You are an expert in diagnosing and fixing software problems')
        
        # Format messages for the Ollama chat API
        messages = [
            {
                "role": "system", 
                "content": f"You are {agent_name}, a {agent_role}. Your goal is to {agent_goal}. {agent_backstory}"
            },
            {
                "role": "user", 
                "content": f"""Task: {task_description}
                
Context:
{context}

Please complete this task to the best of your ability, providing detailed analysis and insights.
"""
            }
        ]
        
        # Call Ollama directly
        logger.info(f"Executing task '{task_description}' directly with Ollama")
        result = LLMProvider.ollama_chat_completion(
            messages=messages,
            model=model
        )
        
        return result
    
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
        
        # Special handling for Ollama provider - bypass CrewAI
        if self.provider == 'ollama':
            logger.info("Using direct Ollama API execution to bypass CrewAI integration")
            
            # Execute each task in sequence and collect results
            task_results = []
            task_outputs = []
            
            for task in tasks:
                # Get agent name safely
                try:
                    agent_name = task.agent.name
                except AttributeError:
                    agent_name = getattr(task.agent, 'name', f"Agent_{tasks.index(task)+1}")
                
                logger.info(f"Running task for agent {agent_name}")
                
                # Update context with previous results
                context_with_results = context.copy()
                if task_outputs:
                    context_with_results.append({
                        "previous_results": task_outputs
                    })
                
                # Execute the task directly with Ollama
                result = self.execute_direct_ollama_task(task, context_with_results)
                
                # Store the result
                task_results.append({
                    "agent": agent_name,
                    "description": task.description,
                    "result": result
                })
                task_outputs.append(result)
                
                logger.info(f"Completed task for agent {agent_name}")
            
            # Generate final output string
            crew_output = "\n\n".join([
                f"## {result['agent']} Results:\n{result['result']}"
                for result in task_results
            ])
        else:
            # Use regular CrewAI for other providers
            # Create crew with provider config
            crew_config = {
                "agents": [a["crew_agent"] for a in self.agents],
                "tasks": tasks,
                "process": Process.sequential,
                "verbose": True
            }
            
            # Add provider override if needed
            if self.provider == 'bedrock' or self.provider == 'anthropic':
                crew_config["llm_provider"] = "bedrock"
            elif self.provider == 'ollama':
                # Use OpenAI provider but with our configured base URL
                crew_config["llm_provider"] = "openai"
                
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