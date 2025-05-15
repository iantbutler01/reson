"""
Example demonstrating the parser integrations for GASP and BAML.

This example shows how to use the parser framework with both
streaming and non-streaming calls, with both GASP and BAML.
"""

import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel

from reson.reson import agentic, Runtime


# Define output types
class Person(BaseModel):
    """A person with name and age."""
    name: str
    age: int
    skills: List[str] = []


class Company(BaseModel):
    """A company with name, domain, and employees."""
    name: str
    domain: str
    employees: List[Person]
    ceo: Optional[Person] = None


# Example using GASP parsing
@agentic(model="openrouter:openai/gpt-4o")
async def extract_people(people_list: List[str], runtime: Runtime) -> List[Person]:
    """
    Extract people information from the following descriptions:
    
    {% for person_desc in people_list %}
    Person {{ loop.index }}: {{ person_desc }}
    {% endfor %}
    
    Return a list of Person objects with names, ages, and skills.
    
    {{return_type}}
    """
    # Debug: Print expected return type
    print(f"Output type: {List[Person]}")
    
    # Use the enhanced prompt with Jinja2 template processing
    result = await runtime.run()
    print(f"Result type: {type(result).__name__}")
    if result and len(result) > 0:
        print(f"First item type: {type(result[0]).__name__}")
    return result


@agentic(model="openrouter:openai/gpt-4o")
async def extract_company(prompt: str, runtime: Runtime) -> Company:
    """
    Extract company information from the given text.
    Return a Company object with name, domain, and employees.
    """
    # Manual approach for customized prompts with complete control
    # Here we're building a prompt with additional instructions
    # while still including the {{return_type}} placeholder
    enhanced_prompt = f"""
    Here's a company description that needs parsing:
    
    {prompt}
    
    Please extract all the information about the company, including:
    - Company name and domain
    - All employees mentioned and their skills
    - CEO information if available
    
    {{{{return_type}}}}
    """
    
    # Streaming call with typed output
    result: Company
    async for chunk in runtime.run_stream(prompt=enhanced_prompt):
        result = chunk
        # In a real app, you might update UI with each chunk
        print(f"Received update: {chunk}")

    print(type(result))
    
    return result


# BAML integration example
@agentic(model="anthropic:claude-3-opus-20240229")
async def extract_with_baml(runtime):
    """Use BAML integration to extract information."""
    try:
        # Check if BAML is available
        import baml_client as b
        
        # Create a BAML request
        request = b.ExtractPerson("John Doe is a 30-year-old software engineer with skills in Python, JavaScript, and DevOps.")
        
        # Run the request through our runtime
        result = await runtime.run_with_baml(baml_request=request)
        
        return result
    except ImportError:
        print("BAML client not installed. Run: pip install baml-client")
        return Person(name="Example Person", age=0)


@agentic(model="anthropic:claude-3-opus-20240229")
async def extract_with_baml_stream(runtime: Runtime):
    """Use BAML integration to extract information with streaming."""
    try:
        # Check if BAML is available
        import baml_client as b
        
        # Create a BAML request
        request = b.ExtractCompany("""
        Acme Corporation is a technology company in the AI domain.
        Their CEO is John Smith, a 45-year-old visionary with skills in leadership and strategy.
        They have employees like:
        - Jane Doe, 32, skills: Python, ML, DevOps
        - Mike Johnson, 28, skills: JavaScript, React, UI/UX
        - Sarah Williams, 35, skills: Product Management, Agile, Marketing
        """)
        
        # Stream the results
        result = None
        async for chunk in runtime.run_stream_with_baml(baml_request=request):
            result = chunk
            # In a real app, you might update UI with each chunk
            print(f"Received BAML update: {chunk}")
        
        return result
    except ImportError:
        print("BAML client not installed. Run: pip install baml-client")
        return Company(name="Example Corp", domain="example.com", employees=[])


async def main():
    """Run the example."""
    print("Extracting people...")
    people = await extract_people(people_list=[
        "John Smith is a 42-year-old engineer with skills in Python, C++, and AI.",
        "Jane Doe is a 35-year-old designer with skills in UI/UX, Figma, and CSS."
    ])
    print(f"Extracted people: {people}")
    
    print("\nExtracting company (streaming)...")
    company = await extract_company("""
    TechCorp is a technology company in the cloud computing domain.
    They have several employees including:
    - Robert Johnson, 38, skills: Cloud Architecture, AWS, Kubernetes
    - Emily Chen, 29, skills: Data Science, Python, TensorFlow
    - Michael Brown, 45, skills: Project Management, Agile, Leadership
    Their CEO is Lisa Wong, a 52-year-old veteran with skills in strategy and business development.
    """)
    print(f"Final company data: {company}")
    
    try:
        print("\nExtracting using BAML integration...")
        baml_person = await extract_with_baml()
        print(f"BAML extracted person: {baml_person}")
        
        print("\nExtracting using BAML streaming integration...")
        baml_company = await extract_with_baml_stream()
        print(f"BAML final company data: {baml_company}")
    except Exception as e:
        print(f"BAML example error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
