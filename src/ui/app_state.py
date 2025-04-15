"""
State management for the Streamlit UI
"""
from enum import Enum
from typing import Dict, Any, Optional
import streamlit as st
from datetime import datetime

class DebugStage(Enum):
    SETUP = "Setting up debugging environment"
    FORECASTING = "Running forecasting pipeline"
    CONTEXT = "Building debug context"
    ANALYSIS = "Running debug analysis"
    PLAN = "Creating debug plan"
    EXECUTION = "Executing debug plan"
    DOCUMENT = "Generating documentation"
    COMPLETE = "Debug complete"

class AppState:
    """Manages application state using Streamlit's session state"""
    
    def __init__(self):
        """Initialize application state"""
        if 'stage_status' not in st.session_state:
            st.session_state.stage_status = {}
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
        if 'success' not in st.session_state:
            st.session_state.success = False
        if 'jira_id' not in st.session_state:
            st.session_state.jira_id = None
            
    @property
    def error_message(self) -> Optional[str]:
        """Get the current error message"""
        return st.session_state.error_message
            
    @property
    def success(self) -> bool:
        """Get the current success state"""
        return st.session_state.success
            
    def update_stage(self, stage: DebugStage, status: str, message: str = None):
        """Update the status and message for a debug stage"""
        if not isinstance(stage, DebugStage):
            raise ValueError(f"Invalid stage: {stage}")
            
        if status not in ['pending', 'running', 'complete', 'error']:
            raise ValueError(f"Invalid status: {status}")
            
        st.session_state.stage_status[stage] = {
            'status': status,
            'message': message or '',
            'timestamp': datetime.now().isoformat()
        }
        
    def set_error(self, message: str):
        """Set error message and update all incomplete stages to error"""
        st.session_state.error_message = message
        st.session_state.success = False
        
        # Update any incomplete stages to error
        for stage in DebugStage:
            if stage not in st.session_state.stage_status or \
               st.session_state.stage_status[stage]['status'] in ['pending', 'running']:
                self.update_stage(stage, 'error', message)
                
    def set_success(self, success: bool):
        """Set the success state"""
        st.session_state.success = success
        
    def reset(self):
        """Reset the application state"""
        st.session_state.stage_status = {}
        st.session_state.error_message = None
        st.session_state.success = False
        
    def get_stage_status(self, stage: DebugStage) -> dict:
        """Get the status info for a specific stage"""
        if stage not in st.session_state.stage_status:
            return {
                'status': 'pending',
                'message': '',
                'timestamp': None
            }
        return st.session_state.stage_status[stage]
        
    def is_stage_complete(self, stage: DebugStage) -> bool:
        """Check if a stage is complete"""
        status = self.get_stage_status(stage)
        return status['status'] == 'complete'
        
    def is_stage_error(self, stage: DebugStage) -> bool:
        """Check if a stage has errored"""
        status = self.get_stage_status(stage)
        return status['status'] == 'error'
        
    def is_stage_running(self, stage: DebugStage) -> bool:
        """Check if a stage is currently running"""
        status = self.get_stage_status(stage)
        return status['status'] == 'running'
        
    def is_any_stage_running(self) -> bool:
        """Check if any stage is currently running"""
        return any(
            self.is_stage_running(stage)
            for stage in DebugStage
        )

    def get_stage_message(self, stage: DebugStage) -> str:
        """Get the message for a specific stage"""
        status = self.get_stage_status(stage)
        return status.get('message', '')

    @staticmethod
    def initialize():
        """Initialize all state variables"""
        if 'debug_state' not in st.session_state:
            st.session_state.debug_state = 'ready'
        if 'doc_state' not in st.session_state:
            st.session_state.doc_state = 'idle'
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = None
        if 'abort_requested' not in st.session_state:
            st.session_state.abort_requested = False
        if 'doc_generated' not in st.session_state:
            st.session_state.doc_generated = False
    
    @staticmethod
    def request_abort():
        """Request abortion of the debugging process"""
        st.session_state.abort_requested = True
        st.session_state.debug_state = 'ready'
        st.session_state.current_stage = None
        st.session_state.stage_status = {}
        st.session_state.doc_generated = False
    
    @staticmethod
    def start_debug(jira_id: str):
        """Start debugging process"""
        st.session_state.debug_state = 'running'
        st.session_state.abort_requested = False
        st.session_state.jira_id = jira_id
        st.session_state.error_message = None
        st.session_state.stage_status = {}
        st.session_state.doc_generated = False 