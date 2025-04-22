"""
Debug Crew Manager - Manages crew of debugging agents.
"""

import os
import sys
import json
import logging
import httpx
import boto3
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
        
        # Create LLM from factory (or directly for bedrock)
        self.llm = LLMFactory.create_llm(provider=self.provider)
        
        # Determine if we need to use direct API calls
        if self.provider == 'bedrock':
            logger.info("Using direct Bedrock API integration")
            self.use_direct_api = False
            self.use_direct_bedrock = True
        elif self.provider == 'ollama':
            logger.info("Using direct Ollama API integration")
            self.use_direct_api = True
            self.use_direct_bedrock = False
        else:
            self.use_direct_api = False
            self.use_direct_bedrock = False
        
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
                           f"with years of experience diagnosing and fixing complex issues."
            }
            
            # Only add the LLM to the agent config if we're not using direct API
            if not self.use_direct_api:
                agent_config["llm"] = self.llm
            
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
                    if line:  # Skip empty lines
                        try:
                            response_part = json.loads(line)
                            response_text = response_part.get("response", "")
                            all_text += response_text
                            
                            # Check for done flag
                            if response_part.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode Ollama response line: {line}. Error: {str(e)}")
                
                if not all_text:
                    logger.warning("Received empty response from Ollama")
                    return "No response received from Ollama"
                    
                logger.debug(f"Ollama response length: {len(all_text)} chars")
                return all_text
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Ollama API: {str(e)}")
            return f"HTTP Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {str(e)}")
            return f"Error: {str(e)}"
    
    def call_bedrock_directly(self, prompt, system_message=None):
        """
        Call Bedrock API directly without using any abstractions.
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            Generated text
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Call Bedrock API directly
        # Default to environment variables if not provided
        model = os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
        region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        temperature = 0.2
        
        logger.info(f"Calling AWS Bedrock directly with model={model}")
        
        # Set up AWS session
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        session_kwargs = {}
        if aws_access_key and aws_secret_key:
            session_kwargs = {
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key,
                'region_name': region
            }
        
        # Create Bedrock client
        session = boto3.Session(**session_kwargs)
        client = session.client('bedrock-runtime', region_name=region)
        
        try:
            # Extract system and user messages
            system_message = None
            user_message = None
            
            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                elif msg.get("role") == "user":
                    user_message = msg.get("content", "")
            
            # Create the request payload for Claude
            anthropic_messages = []
            
            if system_message:
                anthropic_messages.append({
                    "role": "system", 
                    "content": system_message
                })
            
            if user_message:
                anthropic_messages.append({
                    "role": "user",
                    "content": user_message
                })
            
            # Create request body for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": anthropic_messages
            }
            
            # Invoke the model
            response = client.invoke_model(
                modelId=model,
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read().decode('utf-8'))
            
            return response_body.get('content', [{}])[0].get('text', '')
        except Exception as e:
            logger.error(f"Error calling Bedrock API: {str(e)}")
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
    
    def run_direct_bedrock(self, issue_id: str) -> Dict[str, Any]:
        """
        Run the debugging process directly using Bedrock API
        
        Args:
            issue_id: The issue ID to debug
            
        Returns:
            Dict with results information
        """
        logger.info(f"Running direct Bedrock debugging for issue {issue_id}")
        
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
            
            # Call Bedrock directly
            result = self.call_bedrock_directly(
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
        final_summary = self.call_bedrock_directly(
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
            # If using direct API implementations
            if self.use_direct_api:
                logger.info("Using direct Ollama API implementation")
                result = self.run_direct_ollama(issue_id)
                return {"crew_output": result, "document_url": ""}
            
            # If using direct Bedrock implementation
            if self.use_direct_bedrock:
                logger.info("Using direct Bedrock API implementation")
                result = self.run_direct_bedrock(issue_id)
                
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
        <pre>{result}</pre>
    </div>
</body>
</html>""")
                
                logger.info(f"Report generated at {report_path}")
                
                return {
                    "crew_output": result,
                    "document_url": report_path
                }
            
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
