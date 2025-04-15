"""
Template Manager - Handles RCA report templates.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages RCA report templates."""
    
    def __init__(self, template_dir: str = "data/templates"):
        """
        Initialize the template manager.
        
        Args:
            template_dir: Directory containing templates (defaults to data/templates)
        """
        self.template_dir = template_dir
        self.templates: Dict[str, Dict] = {}
        
        logger.info(f"Initialized TemplateManager with template directory: {self.template_dir}")
    
    def load_template(self, template_name: str) -> Dict:
        """Load a template from the template directory."""
        if template_name in self.templates:
            return self.templates[template_name]

        template_path = os.path.join(self.template_dir, template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r') as f:
            template = json.load(f)
            self.templates[template_name] = template
            return template
    
    def apply_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """Apply data to a template and return the formatted content."""
        template = self.load_template(template_name)
        content = []

        # Add title
        content.append(template["title"])
        content.append("\n" + "=" * len(template["title"]) + "\n")

        # Process each section
        for section in template["sections"]:
            content.append(f"\n{section['name']}")
            content.append("-" * len(section['name']))
            
            # Format the section template with the provided data
            section_content = section["template"].format(**data)
            content.append(section_content)

        # Add footer
        content.append(f"\n\n{template['footer']}")

        return "\n".join(content)

    def save_report(self, content: str, output_path: str) -> str:
        """Save the generated report to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path

    def _load_json_template(self, template_path: str) -> Dict[str, Any]:
        """
        Load a JSON template.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Template data as dictionary
        """
        try:
            with open(template_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON template: {e}")
            raise
    
    def _load_docx_template(self, template_path: str) -> Dict[str, Any]:
        """
        Load a DOCX template.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Template data as dictionary
        """
        try:
            doc = docx.Document(template_path)
            
            # Extract template structure
            template = {
                "sections": [],
                "placeholders": []
            }
            
            # Process each paragraph
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    # Check for section headers
                    if para.style.name.startswith('Heading'):
                        template["sections"].append({
                            "title": text,
                            "level": int(para.style.name[-1]),
                            "content": []
                        })
                    else:
                        # Check for placeholders
                        if "{{" in text and "}}" in text:
                            template["placeholders"].append({
                                "text": text,
                                "placeholder": text[text.find("{{"):text.find("}}")+2]
                            })
                        else:
                            # Add to last section or create new one
                            if template["sections"]:
                                template["sections"][-1]["content"].append(text)
                            else:
                                template["sections"].append({
                                    "title": "Content",
                                    "level": 1,
                                    "content": [text]
                                })
            
            return template
            
        except Exception as e:
            logger.error(f"Error loading DOCX template: {e}")
            raise
    
    def _load_pdf_template(self, template_path: str) -> Dict[str, Any]:
        """
        Load a PDF template.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Template data as dictionary
        """
        try:
            with open(template_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                
                template = {
                    "sections": [],
                    "placeholders": []
                }
                
                # Process each page
                for page in pdf.pages:
                    text = page.extract_text()
                    
                    # Split into lines and process
                    for line in text.split('\n'):
                        line = line.strip()
                        if line:
                            # Check for placeholders
                            if "{{" in line and "}}" in line:
                                template["placeholders"].append({
                                    "text": line,
                                    "placeholder": line[line.find("{{"):line.find("}}")+2]
                                })
                            else:
                                # Add to last section or create new one
                                if template["sections"]:
                                    template["sections"][-1]["content"].append(line)
                                else:
                                    template["sections"].append({
                                        "title": "Content",
                                        "level": 1,
                                        "content": [line]
                                    })
                
                return template
                
        except Exception as e:
            logger.error(f"Error loading PDF template: {e}")
            raise 