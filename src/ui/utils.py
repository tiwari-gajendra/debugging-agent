"""
Utility functions for UI components
"""
import os
from pathlib import Path
from typing import Dict
import streamlit as st
from jinja2 import Environment, FileSystemLoader, select_autoescape

class UILoader:
    """Handles loading of UI templates and styles"""
    
    _templates: Dict[str, str] = {}
    _styles: Dict[str, str] = {}
    _base_path = Path(__file__).parent
    _env = None
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if UILoader._env is None:
            template_path = self._base_path / "templates"
            UILoader._env = Environment(
                loader=FileSystemLoader(template_path),
                autoescape=select_autoescape()
            )

    @classmethod
    def load_template(cls, template_name: str) -> str:
        """
        Load an HTML template file.
        
        Args:
            template_name: Name of the template file (without .html extension)
            
        Returns:
            Template content as string
        """
        if template_name not in cls._templates:
            template_path = cls._base_path / 'templates' / f'{template_name}.html'
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: {template_name}")
            
            with open(template_path, 'r') as f:
                cls._templates[template_name] = f.read()
        
        return cls._templates[template_name]
    
    @classmethod
    def load_style(cls, style_name: str) -> str:
        """
        Load a CSS style file.
        
        Args:
            style_name: Name of the style file (without .css extension)
            
        Returns:
            Style content as string
        """
        if style_name not in cls._styles:
            style_path = cls._base_path / 'styles' / f'{style_name}.css'
            if not style_path.exists():
                raise FileNotFoundError(f"Style not found: {style_name}")
            
            with open(style_path, 'r') as f:
                cls._styles[style_name] = f.read()
        
        return cls._styles[style_name]
    
    @classmethod
    def apply_style(cls, style_name: str):
        """
        Apply a CSS style to the Streamlit app.
        
        Args:
            style_name: Name of the style file (without .css extension)
        """
        style_content = cls.load_style(style_name)
        st.markdown(f"<style>{style_content}</style>", unsafe_allow_html=True)
    
    @classmethod
    def render_template(cls, template_name: str, **kwargs) -> str:
        """
        Render an HTML template with the given parameters.
        
        Args:
            template_name: Name of the template file (without .html extension)
            **kwargs: Template parameters
            
        Returns:
            Rendered template as string
        """
        template = cls.load_template(template_name)
        return template.format(**kwargs)

    @staticmethod
    def apply_style(style_name: str) -> str:
        """Load and return CSS styles from a template file."""
        loader = UILoader.get_instance()
        template = loader._env.get_template(f"styles/{style_name}.css")
        return template.render()

    @staticmethod
    def render_template(template_name: str, **kwargs) -> str:
        """Render a template with the given context."""
        loader = UILoader.get_instance()
        template = loader._env.get_template(f"{template_name}.html")
        return template.render(**kwargs) 