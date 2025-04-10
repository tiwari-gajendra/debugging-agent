from setuptools import setup, find_packages

setup(
    name="debugging-agents",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["debug_agent_cli"],
    scripts=["debug_agent_cli.py"],
    install_requires=[
        # Add your dependencies here
        "requests",
        "langchain",
        "openai",
        "boto3",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "debug-agent=debugging_agents.debug_agent_cli:main",
        ],
    },
    python_requires=">=3.10",
    description="AI-powered debugging and forecasting system",
    author="Debugging Agents Team",
    author_email="debug@example.com",
    url="https://github.com/yourusername/debugging-agents",
) 