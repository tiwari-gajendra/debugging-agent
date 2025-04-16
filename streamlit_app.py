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
                'debug_state': 'ready',  # Add debug state initialization
                'doc_format': 'doc',  # Default document format
                'needs_refresh': False  # Flag to trigger UI refresh
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
            
            # CRITICAL: Set success states in proper order and log each step
            self.logger.info("Debug analysis completed successfully - setting success states")
            
            # First, set is_debugging to False BEFORE anything else
            st.session_state.is_debugging = False
            self.logger.info("is_debugging set to False")
            
            # Then update all success indicators
            st.session_state.debug_state = 'success'
            self.logger.info("debug_state set to 'success'")
            
            st.session_state.app_state.set_success(True)
            self.logger.info("app_state.success set to True")
            
            st.session_state.success = True
            self.logger.info("success flag set to True")
            
            # Set flag to force refresh UI on next cycle
            st.session_state.needs_refresh = True
            self.logger.info("needs_refresh set to True - will trigger rerun")
            
            # Log success
            self.logger.info(f"Debug analysis completed successfully for {jira_id}")
            
            # Force a UI refresh
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error running debug analysis: {str(e)}", exc_info=True)
            st.session_state.app_state.set_error(str(e))
            st.session_state.debug_state = 'error'
            
            # Also ensure is_debugging is set to False on error
            st.session_state.is_debugging = False
            
            raise
    
    def render_ui(self):
        """Render UI components"""
        ui = UIComponents(st.session_state.app_state)
        ui.apply_styles()
        
        # Display header directly instead of calling non-existent show_header method
        st.title("🔍 Agentic Debugger")
        
        st.markdown("### ➡️ Enter JIRA Ticket ID")
        
        # Display validation error in a consistent location
        if 'jira_validation_error' in st.session_state:
            st.error(f"{st.session_state.jira_validation_error}", icon="⚠️")
        
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

        # Clear validation error if valid JIRA entered - but don't add this check inside the layout
        if jira_id and st.session_state.get("jira_validation_error"):
            jira_client = JiraClient()
            validation_check = asyncio.run(jira_client.validate_ticket_exists(jira_id))
            if validation_check.get("exists"):
                del st.session_state.jira_validation_error
                st.rerun()  # Rerun to refresh UI without the error message
        
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
        
        # Add some spacing before the document section
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Show document section, completely separate from validation warnings
        self.show_document_section()
        
        # Handle abort request
        if st.session_state.abort_requested:
            self.logger.warning("Debug process aborted by user")
            st.session_state.app_state.set_error("Debug process aborted by user")
            st.session_state.abort_requested = False
            st.session_state.is_debugging = False
            st.session_state.app_state.reset()
            st.rerun()
        
        return jira_id, start_clicked
    
    def show_document_section(self):
        """Show a single, consolidated document section"""
        # Always show the document section header
        st.markdown("### 📄 Download RCA")
        
        # Create columns for side by side layout (same as before)
        col1, col2 = st.columns([3, 1])
        
        # Format selector in first column
        with col1:
            # Only disable during active debugging, not success state
            is_debugging_active = st.session_state.get('is_debugging', False)
            
            st.selectbox(
                "Document Format",
                options=["doc", "pdf"],
                index=0,
                key="doc_format_selector",  # Use a different key to avoid conflicts
                disabled=is_debugging_active,
            )
            
            # Update the format in session state for use elsewhere
            if "doc_format_selector" in st.session_state:
                st.session_state.doc_format = st.session_state.doc_format_selector
        
        # Download button in second column
        with col2:
            # Add spacing to align with dropdown
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
            
            # If debugging is active, always show disabled button
            is_debugging_active = st.session_state.get('is_debugging', False)
            if is_debugging_active:
                st.button(
                    "📥 Download Not Available",
                    disabled=True,
                    use_container_width=True,
                )
                return
            
            reports_dir = Path("data/reports")
            jira_id = st.session_state.jira_id.strip().upper()
            format = st.session_state.get('doc_format', 'doc')
            
            # Find reports for this JIRA ID
            all_report_files = []
            download_available = False
            
            if jira_id:
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                # First check for all available formats (including html which is the default output)
                for ext in ["html", "doc", "pdf"]:
                    # Look for both prefixes
                    for prefix in ["BIM_", "debug_report_"]:
                        all_report_files.extend(reports_dir.glob(f"{prefix}{jira_id}*.{ext}"))
            
            # Show download button if files exist
            if all_report_files:
                # Get the latest report file, regardless of format
                latest_report = max(all_report_files, key=lambda x: x.stat().st_mtime)
                
                # If the latest report isn't in the requested format, we need to convert it
                if latest_report.suffix[1:] != format:
                    try:
                        # Use the document generator to convert
                        doc_generator = DocumentGenerator()
                        converted_file = doc_generator._convert_format(latest_report, format)
                        if Path(converted_file).exists():
                            latest_report = Path(converted_file)
                            download_available = True
                    except Exception:
                        # Silently handle any conversion errors - don't show technical details
                        download_available = False
                else:
                    download_available = True
                
                # Only try to download if conversion was successful or file already exists in correct format
                if download_available:
                    try:
                        # Read the file after conversion attempt
                        with open(latest_report, 'rb') as f:
                            report_data = f.read()
                        
                        # Set mime type based on final format
                        if format == "doc":
                            mime_type = "application/msword"
                        elif format == "pdf":
                            mime_type = "application/pdf"
                        else:  # default for html
                            mime_type = "text/html"
                        
                        # Show download button with appropriate file name
                        st.download_button(
                            label="📥 Download",
                            data=report_data,
                            file_name=f"{jira_id}_report.{latest_report.suffix[1:]}",
                            mime=mime_type,
                            use_container_width=True,
                        )
                    except Exception:
                        # If any error occurs during file reading or button creation, show not available
                        st.button(
                            "📥 Download Not Available",
                            disabled=True,
                            use_container_width=True,
                        )
                else:
                    # Show not available if conversion failed
                    st.button(
                        "📥 Download Not Available",
                        disabled=True,
                        use_container_width=True,
                    )
            else:
                # Always show a disabled button when no files are available
                st.button(
                    "📥 Download Not Available",
                    disabled=True,
                    use_container_width=True,
                )
        
        # Add spacing after the document section
        st.write("")
        st.write("")
    
    def run(self):
        """Main application loop"""
        try:
            # Initialize session state
            self._initialize_session_state()
            
            # Ensure app_state exists
            if 'app_state' not in st.session_state:
                st.session_state.app_state = AppState()
            
            # Check if we need to force refresh the UI after debugging completes
            if st.session_state.get('needs_refresh', False):
                st.session_state.needs_refresh = False
                st.rerun()
            
            # Get UI values - this handles all UI rendering including document sections
            jira_id, start_clicked = self.render_ui()
            
            if start_clicked and jira_id and not st.session_state.is_debugging:
                st.session_state.app_state.reset()
                st.session_state.is_debugging = True
                st.session_state.jira_id = jira_id  # Store JIRA ID in session state
                
                try:
                    # Validate if the JIRA ticket exists before proceeding
                    jira_client = JiraClient()
                    validation_result = asyncio.run(jira_client.validate_ticket_exists(jira_id))
                    
                    if not validation_result["exists"]:
                        # Show friendly warning but don't proceed with debugging
                        self.logger.warning(f"Invalid ticket: {validation_result['message']}")
                        # Store validation error in session state for UI display
                        st.session_state.jira_validation_error = validation_result['message']
                        st.session_state.is_debugging = False  # Reset debugging state
                        st.rerun()  # Rerun to display the error
                        return  # Exit early without starting debug
                    
                    # Only run debug analysis if ticket exists
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