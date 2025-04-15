"""
UI Components for the Streamlit application
"""
import streamlit as st
from pathlib import Path
from .app_state import AppState, DebugStage
from .utils import UILoader
import time
import os
from jinja2 import Environment, FileSystemLoader

class UIComponents:
    """Handles UI component rendering"""
    
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.template_dir = Path(__file__).parent / "templates"
    
    def _load_template(self, template_name: str) -> str:
        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template {template_name} not found")
        return template_path.read_text()
    
    def apply_styles(self):
        """Apply CSS styles from the base template"""
        loader = UILoader.get_instance()
        css = loader.apply_style('base')
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    @staticmethod
    def show_header():
        """Show application header"""
        st.title("🔍 JIRA Debugging Agents")
        st.markdown("### Enter JIRA Ticket Details")
    
    @staticmethod
    def show_input_section():
        """Show JIRA ticket input section"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            jira_id = st.text_input(
                "JIRA Ticket ID",
                value=st.session_state.jira_id,
                placeholder="e.g., PROJ-123",
                help="Enter a valid JIRA ticket ID to start debugging",
                disabled=st.session_state.debug_state == 'running',
                key="jira_id_input"
            ).strip().upper()
            
            # Update session state if JIRA ID changes
            if jira_id != st.session_state.jira_id:
                st.session_state.jira_id = jira_id
        
        with col2:
            if st.session_state.debug_state == 'running':
                if st.button("🛑 Abort Debug", type="secondary", help="Stop the current debugging process", key="abort_button"):
                    AppState.request_abort()
                    st.rerun()
            elif jira_id:
                if st.button("Start Debug", type="primary", help="Start debugging process", key="start_button"):
                    AppState.start_debug(jira_id)
                    st.rerun()
    
    @staticmethod
    def show_progress():
        """Show debug progress"""
        if st.session_state.abort_requested:
            st.warning("⚠️ Debugging process aborted by user.", icon="⚠️")
            return
        
        if st.session_state.debug_state == 'running':
            st.markdown("""
                <div class="stage-container stage-running">
                    <span class="signal"></span>
                    <span>🔄 Debug in progress...</span>
                </div>
            """, unsafe_allow_html=True)
            UIComponents.show_progress_stages()
        elif st.session_state.debug_state == 'error':
            st.error("❌ Debug failed", icon="❌")
            if st.session_state.error_message:
                st.error(st.session_state.error_message)
        elif st.session_state.debug_state == 'success':
            st.success("✅ Debug completed successfully", icon="✅")
    
    @staticmethod
    def show_document_section():
        """Show document generation section"""
        st.markdown("### 📄 Download RCA")
        
        # Create columns for side by side layout
        col1, col2 = st.columns([3, 1])
        
        # Format selector
        with col1:
            doc_format = st.selectbox(
                "Document Format",
                options=["doc", "pdf", "markdown"],
                index=0,
                help="Select the output format for the BIM document",
                key="doc_format",
                disabled=st.session_state.debug_state == 'running'
            )
        
        # Download button
        with col2:
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  # More precise spacing
            if st.session_state.debug_state == 'success':
                reports_dir = Path("data/reports")
                jira_id = st.session_state.jira_id.upper()
                report_files = []
                
                for pattern in [f"BIM_{jira_id}*", f"debug_report_{jira_id}*"]:
                    report_files.extend(list(reports_dir.glob(pattern)))
                
                if report_files:
                    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_report, 'rb') as f:
                        st.download_button(
                            label="📥 Download",
                            data=f,
                            file_name=f"{jira_id}_report.{doc_format}",
                            mime=f"application/{doc_format}",
                            help=f"Download the generated document in {doc_format.upper()} format",
                            key="download_button",
                            use_container_width=True
                        )
                else:
                    st.button(
                        "📥 Download",
                        disabled=True,
                        help="No report files found",
                        key="download_button_missing",
                        use_container_width=True
                    )
            else:
                st.button(
                    "📥 Download",
                    disabled=True,
                    help="Start debugging to generate a downloadable document",
                    key="download_button_ready",
                    use_container_width=True
                )
        
        # Show status messages below the controls
        if st.session_state.debug_state == 'running':
            st.info("🔄 Debug in progress... Document will be available once completed.", icon="🔄")
        elif st.session_state.debug_state == 'error':
            st.error("❌ Debug failed. Please fix errors and try again.", icon="❌")
        elif st.session_state.debug_state == 'success':
            if not report_files:
                st.warning("⚠️ No report files found. Please try regenerating the document.", icon="⚠️")
            else:
                st.success("✅ Debug completed successfully. You can now download the document.", icon="✅")
    
    def show_progress_stages(self):
        """Show detailed progress stages with status indicators"""
        for stage in DebugStage:
            stage_info = self.app_state.get_stage_status(stage)
            status = stage_info['status']
            message = stage_info['message']
            
            # Determine stage status and styling
            signal_class = f"signal signal-{status}"
            stage_class = f"stage-{status}"
            
            icon = {
                'running': "🔄",
                'complete': "✅",
                'error': "❌",
                'pending': "⏳"
            }.get(status, "⏳")
            
            # Show stage with signal indicator and message
            st.markdown(f"""
                <div class="progress-stage {stage_class}">
                    <div style="display: flex; align-items: center;">
                        <span class="{signal_class}"></span>
                        <span style="margin-right: 8px;">{icon}</span>
                        <span>{stage.value}</span>
                    </div>
                    {f'<div style="margin-left: 26px; font-style: italic; color: inherit;">{message}</div>' if message else ''}
                </div>
            """, unsafe_allow_html=True)
            
            # Add a small spacing between stages
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            
            # Only rerun if the stage is actually running and not aborted
            if status == 'running' and not self.app_state.is_aborted():
                time.sleep(0.1)  # Small delay to prevent too frequent updates
                st.rerun()
    
    @staticmethod
    def show_status_box(state: str, message: str):
        """Show a styled status box"""
        css_class = {
            'running': 'running',
            'success': 'success',
            'error': 'error'
        }.get(state, '')
        
        html = UILoader.render_template('status_box',
            css_class=css_class,
            message=message
        )
        st.markdown(html, unsafe_allow_html=True)
    
    def show_error(self, error_message: str):
        """Display an error message"""
        st.error(error_message)
    
    def show_success(self, success_message: str):
        """Display a success message"""
        st.success(success_message)
    
    def show_abort_button(self):
        """Display and handle the abort button"""
        if st.button("🛑 Abort Debug", type="secondary"):
            self.app_state.set_abort_flag()
            st.rerun() 