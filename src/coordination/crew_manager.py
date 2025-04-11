"""
Debug Crew Manager - Manages crew of debugging agents.
"""

import os
import sys
import json
import logging
import httpx
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
        
        # If using Ollama, create a pre-configured LLM instance that works with CrewAI
        if self.provider == 'ollama':
            # Get Ollama settings
            model_name = os.getenv('OLLAMA_MODEL', 'deepseek-r1:8b')
            # Remove 'ollama/' prefix if present
            if model_name.startswith('ollama/'):
                model_name = model_name[len('ollama/'):]
            
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
            # For CrewAI compatibility
            os.environ["OPENAI_API_KEY"] = "sk-valid-ollama-key"
            
            # Import CrewAI LLM to create a pre-configured instance
            try:
                from crewai.llm import LLM
                self.llm = LLM(
                    model="ollama/"+model_name,     # Format expected by LiteLLM
                    base_url=base_url,              # Ollama API
                    api_key="sk-valid-ollama-key",  # Placeholder key
                    temperature=float(os.getenv('TEMPERATURE', 0.2)),
                    custom_llm_provider="ollama"    # Force provider type
                )
                logger.info(f"Created pre-configured CrewAI LLM for Ollama with model={model_name}")
                self.use_direct_api = False         # Use standard CrewAI flow now
            except ImportError:
                logger.warning("Could not import crewai.llm, falling back to direct API implementation")
                # Create an LLM instance using our utility
                self.llm = LLMProvider.create_llm(provider_or_model=self.provider)
                self.use_direct_api = True          # Still use direct API as fallback
        else:
            # Create an LLM instance using the unified LLMProvider
            self.llm = LLMProvider.create_llm(provider_or_model=self.provider)
            self.use_direct_api = False
        
        # Set up environment variables for CrewAI compatibility
        if self.provider == 'bedrock' or self.provider == 'anthropic':
            os.environ["OPENAI_API_KEY"] = "sk-valid-bedrock-key"
            os.environ["CREW_LLM_PROVIDER"] = "bedrock"
        elif self.provider == 'ollama':
            os.environ["OPENAI_API_KEY"] = "sk-valid-ollama-key"
            os.environ["CREW_LLM_PROVIDER"] = "ollama"
        
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
            agent = Agent(**agent_config)
            
            # Store both the crewAI agent and the original agent object
            self.agents.append({
                "crew_agent": agent,
                "agent_obj": agent_obj,
                "agent_name": agent_name  # Store name explicitly for logging
            })
            
            logger.debug(f"Added agent {agent_name} to crew")
    
    def call_ollama_directly(self, prompt, system_message=None):
        """
        Call Ollama API directly without using any abstractions.
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            Generated text
        """
        # Get model from environment or use default
        model = os.getenv('OLLAMA_MODEL', 'deepseek-r1:8b')
        
        # Clean the model name - remove ollama/ prefix if present
        if model.startswith('ollama/'):
            model = model[len('ollama/'):]
            
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Remove trailing slash from base_url if present
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # Format the request
        api_url = f"{base_url}/api/generate"
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.2,
        }
        
        # Add system prompt if provided
        if system_message:
            request_data["system"] = system_message
        
        try:
            logger.info(f"Calling Ollama API directly with model={model}")
            
            # Make the API call
            with httpx.Client(timeout=60.0) as client:
                response = client.post(api_url, json=request_data)
                response.raise_for_status()
                
                # Ollama's generate endpoint returns streaming responses
                # We'll collect all the text from the response
                all_text = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        response_part = json.loads(line)
                        response_text = response_part.get("response", "")
                        all_text += response_text
                        
                        # Check for done flag
                        if response_part.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode Ollama response line: {line}")
                
                logger.debug(f"Ollama response length: {len(all_text)} chars")
                return all_text
        
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_direct_ollama(self, issue_id: str) -> Dict[str, Any]:
        """
        Run the debugging process directly using Ollama API
        
        Args:
            issue_id: The issue ID to debug
            
        Returns:
            Dict with results information
        """
        logger.info(f"Running direct Ollama debugging for issue {issue_id}")
        
        # Create context to send to the model
        context = {
            "issue_id": issue_id,
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": sys.version,
                "os": sys.platform
            }
        }
        
        # Use our agent descriptions to create task list
        task_descriptions = []
        for agent_pair in self.agents:
            agent_obj = agent_pair["agent_obj"]
            agent_name = agent_pair["agent_name"]
            
            # Get the task description from agent if available
            task_description = f"Analyze and process the issue {issue_id}"
            if hasattr(agent_obj, "get_task_description"):
                task_description = agent_obj.get_task_description(issue_id)
                
            task_descriptions.append({
                "agent_name": agent_name,
                "description": task_description
            })
        
        # Execute each task in sequence
        task_results = []
        previous_results = []
        
        for i, task in enumerate(task_descriptions):
            agent_name = task["agent_name"]
            task_description = task["description"]
            
            logger.info(f"Running task {i+1}/{len(task_descriptions)}: {agent_name}")
            
            # Create system message for the agent
            system_message = f"""You are {agent_name}, an expert in software debugging and analysis.
Your goal is to provide detailed technical expertise to solve complex software issues.
Respond with detailed, actionable information."""
            
            # Create prompt with task description and context
            prompt = f"""# Task: {task_description}

## Context Information:
Issue ID: {issue_id}

"""
            
            # Add previous results if any
            if previous_results:
                prompt += "\n## Previous Analysis Results:\n"
                for prev in previous_results:
                    prompt += f"\n### {prev['agent']} Results:\n{prev['result']}\n"
            
            # Call Ollama directly
            result = self.call_ollama_directly(
                prompt=prompt,
                system_message=system_message
            )
            
            # Store the result
            task_result = {
                "agent": agent_name,
                "description": task_description,
                "result": result
            }
            task_results.append(task_result)
            previous_results.append(task_result)
            
            logger.info(f"Completed task: {agent_name}")
        
        # Generate final summary
        system_message = """You are a senior software engineer with expertise in debugging and technical documentation.
Your task is to create a comprehensive executive summary of a debugging report."""

        prompt = f"""# Debug Report Summary Request

## Issue Information:
Issue ID: {issue_id}

## Task Results:
"""
        for result in task_results:
            prompt += f"\n### {result['agent']} Results:\n{result['result']}\n"
            
        prompt += """
# Request:
Please provide a concise executive summary of the debugging process and findings.
Include key insights, root causes identified, and recommendations."""

        # Generate final summary
        final_summary = self.call_ollama_directly(
            prompt=prompt,
            system_message=system_message
        )
        
        # Generate final output string
        crew_output = "\n\n".join([
            f"## {result['agent']} Results:\n{result['result']}"
            for result in task_results
        ])
        
        crew_output += f"\n\n## Executive Summary:\n{final_summary}"
        
        return crew_output
    
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
        
        # Use direct API implementation if needed
        if self.use_direct_api:
            logger.info("Running with direct API implementation to avoid LiteLLM issues")
            crew_output = self.run_direct_ollama(issue_id)
        else:
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
                crew_config["llm_provider"] = "ollama"
                
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