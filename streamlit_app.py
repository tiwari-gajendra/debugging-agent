"""
Streamlit UI for JIRA Ticket BIM Document Generation and Debugging

This module provides a web interface for:
1. Searching JIRA tickets
2. Generating BIM documents
3. Running debugging analysis
"""

import sys
import os
import logging
from pathlib import Path
import streamlit as st
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# Core imports
from src.manager.crew_manager import DebugCrew
from src.integrations.jira_client import JiraClient
from src.realtime.document_generator import DocumentGenerator
from src.ui.app_state import AppState, DebugStage
from src.ui.components import UIComponents
from src.ui.utils import UILoader

# Agent imports
from src.realtime.context_builder import ContextBuilder
from src.realtime.debug_plan_creator import DebugPlanCreator
from src.realtime.executor import Executor
from src.realtime.analyzer import Analyzer

class DebugAssistant:
    def __init__(self, debug_timeout_seconds: int = 300):  # 5 minutes default
        self.logger = self._setup_logging()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.debug_timeout = debug_timeout_seconds
        self._initialize_app()
    
    def _setup_logging(self):
        """Configure logging with optimized settings"""
        log_dir = Path("data/logs/debug_agent")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing handlers from the root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.INFO)
        
        # Create formatters and handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_dir / "debug_agent.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Console handler for application logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        # Suppress third-party logs
        for logger_name in ['urllib3', 'streamlit', 'litellm', 'LiteLLM']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        # Create and return the application logger
        app_logger = logging.getLogger("debug_assistant")
        return app_logger
    
    def _initialize_app(self):
        """Initialize application settings"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        st.set_page_config(
            page_title="Debug Assistant",
            page_icon="🔍",
            layout="wide"
        )
    
    def _initialize_session_state(self):
        """Initialize or reset session state"""
        if 'initialized' not in st.session_state:
            # Initialize AppState first
            st.session_state.app_state = AppState()
            
            # Initialize other session state variables
            defaults = {
                'is_debugging': False,
                'abort_requested': False,
                'initialized': True,
                'jira_id': '',  # Store the actual JIRA ID value
                'debug_timeout': self.debug_timeout  # Store timeout in session state
            }
            for key, value in defaults.items():
                st.session_state[key] = value
            self.logger.info(f"Debug Assistant initialized with {self.debug_timeout}s timeout")
    
    @contextmanager
    def error_boundary(self, error_message="An error occurred"):
        """Context manager for error handling"""
        try:
            yield
        except Exception as e:
            self.logger.error(f"{error_message}: {str(e)}", exc_info=True)
            st.error(f"{error_message}: {str(e)}")
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            st.session_state.is_debugging = False
            st.session_state.abort_requested = False
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
    
    async def run_debug_analysis(self, jira_id: str, doc_format: str, app_state: AppState):
        """Run the debug analysis process"""
        try:
            # Initialize the crew
            crew = DebugCrew()
            
            # Initialize agents
            context_builder = ContextBuilder()
            plan_creator = DebugPlanCreator()
            executor = Executor()
            analyzer = Analyzer()
            doc_generator = DocumentGenerator()
            
            # Add agents to crew
            crew.add_agents([
                context_builder,
                plan_creator,
                executor,
                analyzer,
                doc_generator
            ])
            
            # Run the debug process
            result = await crew.run(jira_id)
            
            # Set success state in multiple places to ensure persistence
            st.session_state.debug_state = 'success'
            st.session_state.app_state.success = True
            st.session_state.app_state.set_success(True)
            st.session_state.success = True
            
            # Log success
            self.logger.info(f"Debug analysis completed successfully for {jira_id}")
            
            # Force a UI refresh
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error running debug analysis: {str(e)}", exc_info=True)
            st.session_state.app_state.set_error(str(e))
            st.session_state.debug_state = 'error'
            raise
    
    def render_ui(self):
        """Render UI components"""
        ui = UIComponents(st.session_state.app_state)
        ui.apply_styles()
        UIComponents.show_header()
        
        st.markdown("### ➡️ Enter JIRA Ticket ID")
        
        # Handle JIRA ID input state - only lock when debug is actually running
        if st.session_state.is_debugging or st.session_state.app_state.success:
            # Show readonly text display when debug is actually running
            jira_id = st.text_input(
                "JIRA ID",
                value=st.session_state.jira_id,
                disabled=True,
                help="JIRA ID is locked during/after debugging",
                key="jira_id_display"
            )
        else:
            # Allow input when not actively debugging
            jira_id = st.text_input(
                "JIRA ID",
                value=st.session_state.jira_id,
                placeholder="Enter JIRA ticket ID (e.g., PROJ-123)",
                help="Enter the JIRA ticket ID to analyze",
                key="jira_id_input"
            ).strip().upper()
            # Update session state through direct assignment
            st.session_state.jira_id = jira_id
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col2:
            start_clicked = st.button(
                "Start Debug",
                type="primary",
                use_container_width=True,
                disabled=not jira_id or st.session_state.is_debugging,
                key="start_debug_button"
            )
        
        with col3:
            abort_clicked = st.button(
                "🛑 Abort Debug",
                type="secondary",
                use_container_width=True,
                key="abort_debug_button"
            )
        
        if abort_clicked:
            self.logger.warning("Abort requested by user")
            st.session_state.abort_requested = True
            self.cleanup()
            st.rerun()
        
        return jira_id, start_clicked
    
    def handle_document_config(self):
        """Handle document configuration and document download"""
        if not st.session_state.app_state.success:
            return None
        
        st.markdown("### ➡️ Configure Document Generation")
        st.markdown("Select the output format for your debug document:")
        doc_format = st.selectbox(
            "Document Format",
            options=["doc", "md", "pdf"],
            index=0,
            help="Choose the format for your debug document:\n- DOC: Microsoft Word format\n- MD: Markdown format\n- PDF: Portable Document Format",
            key="doc_format_select"
        )

        # Add download section
        jira_id = st.session_state.jira_id.strip().upper()
        if jira_id:
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to find the report with any extension first
            all_report_files = []
            for ext in ["doc", "md", "pdf"]:
                # Look for both BIM_ and debug_report_ prefixes
                all_report_files.extend(reports_dir.glob(f"BIM_{jira_id}*.{ext}"))
                all_report_files.extend(reports_dir.glob(f"debug_report_{jira_id}*.{ext}"))
            
            if all_report_files:
                # Get the most recent file
                latest_report = max(all_report_files, key=lambda x: x.stat().st_mtime)
                
                # If the requested format is different from the existing file
                if latest_report.suffix[1:] != doc_format:
                    st.info(f"Converting document to {doc_format.upper()} format...")
                    # The DocumentGenerator will handle the conversion
                    
                try:
                    with open(latest_report, 'rb') as file:
                        st.download_button(
                            label=f"📥 Download Debug Report ({doc_format.upper()})",
                            data=file,
                            file_name=f"debug_report_{jira_id}.{doc_format}",
                            mime=self._get_mime_type(doc_format),
                            key="download_button",
                            use_container_width=True,
                        )
                        st.caption("Click the button above to download your report")
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
                    self.logger.error(f"Download error: {str(e)}", exc_info=True)
            else:
                st.info("⏳ Document generation in progress. Please wait...")
                self.logger.info(f"No report files found in {reports_dir} for JIRA ID {jira_id}")
        
        return doc_format
    
    def _get_mime_type(self, doc_format):
        """Get the appropriate MIME type for the document format"""
        mime_types = {
            "doc": "application/msword",
            "pdf": "application/pdf",
            "md": "text/markdown"
        }
        return mime_types.get(doc_format, "application/octet-stream")
    
    def run(self):
        """Main application loop"""
        try:
            # Initialize session state
            self._initialize_session_state()
            
            # Ensure app_state exists
            if 'app_state' not in st.session_state:
                st.session_state.app_state = AppState()
            
            jira_id, start_clicked = self.render_ui()
            doc_format = self.handle_document_config()
            
            if start_clicked and jira_id and not st.session_state.is_debugging:
                st.session_state.app_state.reset()
                st.session_state.is_debugging = True
                st.session_state.jira_id = jira_id  # Store JIRA ID in session state
                
                try:
                    asyncio.run(self.run_debug_analysis(jira_id, "doc", st.session_state.app_state))
                except Exception as e:
                    self.logger.error(f"Error running debug analysis: {str(e)}", exc_info=True)
                    st.session_state.app_state.set_error(str(e))
                finally:
                    # Ensure we reset debugging state and rerun
                    st.session_state.is_debugging = False
                    if not st.session_state.app_state.success:
                        st.session_state.app_state.reset()
                    st.rerun()
            
            # Handle abort request
            if st.session_state.abort_requested:
                self.logger.warning("Debug process aborted by user")
                st.session_state.app_state.set_error("Debug process aborted by user")
                st.session_state.abort_requested = False
                st.session_state.is_debugging = False
                st.session_state.app_state.reset()
                st.rerun()
                
        except Exception as e:
            self.logger.error(f"Error in main application: {str(e)}", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")
            self.cleanup()
            if 'app_state' in st.session_state:
                st.session_state.app_state.reset()
            st.rerun()

def main():
    """Application entry point"""
    # Initialize with 10 minute timeout (600 seconds)
    app = DebugAssistant(debug_timeout_seconds=600)
    app.run()

if __name__ == "__main__":
    main()