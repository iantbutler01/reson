"""
Jinja2 utilities for reson.

This module provides easy access to jinja2 templating functionality
with pre-configured environments that include type helpers from gasp.
"""

import jinja2
import json
from typing import Any, Dict, Optional
from gasp.jinja_helpers import (
    create_type_environment,
    format_type_filter,
    type_description_filter,
    type_to_format_instructions,
    render_template as gasp_render_template,
    render_file_template as gasp_render_file_template
)

# Re-export jinja2 and key functions
__all__ = [
    'jinja2',
    'render_template',
    'render_file_template',
    'create_environment',
    'format_type_filter',
    'type_description_filter',
    'type_to_format_instructions'
]


def render_template(template_str: str, **kwargs: Any) -> str:
    """
    Render a Jinja2 template string with the provided context.
    
    This function creates a Jinja2 environment with type helpers and
    renders the template with the given keyword arguments.
    
    Args:
        template_str: The Jinja2 template string to render
        **kwargs: Context variables to pass to the template
        
    Returns:
        The rendered template string
        
    Example:
        >>> from reson.utils.jinja import render_template
        >>> result = render_template("Hello {{ name }}!", name="World")
        >>> print(result)
        Hello World!
    """
    env = create_type_environment()
    # Add json filter if not already present
    if 'json' not in env.filters:
        env.filters['json'] = lambda obj: json.dumps(obj, indent=2)
    
    template = env.from_string(template_str)
    return template.render(**kwargs)


def render_file_template(template_path: str, **kwargs: Any) -> str:
    """
    Render a Jinja2 template from a file with the provided context.
    
    This is a convenience wrapper around gasp's render_file_template.
    
    Args:
        template_path: Path to the template file
        **kwargs: Context variables to pass to the template
        
    Returns:
        The rendered template string
    """
    return gasp_render_file_template(template_path, **kwargs)


def create_environment(**options: Any) -> jinja2.Environment:
    """
    Create a Jinja2 environment with type helpers pre-configured.
    
    This environment includes filters for working with Python types:
    - format_type: Format a type for display
    - type_description: Get a description of a type
    - type_to_format_instructions: Convert a type to format instructions
    - json: Serialize objects to JSON
    
    Args:
        **options: Additional options to pass to the Jinja2 Environment
        
    Returns:
        A configured Jinja2 Environment
        
    Example:
        >>> from reson.utils.jinja import create_environment
        >>> env = create_environment()
        >>> template = env.from_string("Type: {{ my_type | format_type }}")
        >>> result = template.render(my_type=str)
    """
    env = create_type_environment(**options)
    
    # Ensure json filter is available
    if 'json' not in env.filters:
        env.filters['json'] = lambda obj: json.dumps(obj, indent=2)
    
    return env
