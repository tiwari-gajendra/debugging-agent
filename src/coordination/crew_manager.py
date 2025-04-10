"""
Debug Crew Manager - Manages crew of debugging agents.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from datetime import datetime

from src.utils.llm_factory import LLMFactory

# Load environment variables
load_dotenv()

class DebugCrew:
    """Manages a crew of specialized debugging agents."""
    
    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the Debug Crew.
        
        Args:
            llm_provider: The LLM provider to use (defaults to env var LLM_PROVIDER)
        """
        self.provider = llm_provider or os.getenv('LLM_PROVIDER', 'openai')
        self.llm = LLMFactory.create_llm(self.provider)
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
            
            # Create a crewAI Agent
            agent = Agent(
                name=agent_name,
                role=f"{agent_role} Specialist",
                goal=f"Provide expert {agent_role.lower()} support for debugging issues",
                backstory=f"You are an expert in {agent_role.lower()} for software systems, "
                         f"with years of experience diagnosing and fixing complex issues.",
                llm=self.llm
            )
            
            # Store both the crewAI agent and the original agent object
            self.agents.append({
                "crew_agent": agent,
                "agent_obj": agent_obj
            })
    
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
        
        # Create the crew with sequential process
        crew = Crew(
            agents=[a["crew_agent"] for a in self.agents],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Run the crew
        crew_output = crew.kickoff()
        
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
        
        return result 