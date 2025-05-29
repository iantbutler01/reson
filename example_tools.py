"""
Example demonstrating Pattern 2 - Tool Types functionality.

This example shows how tools are automatically converted to typed
classes based on the output type (Pydantic or Deserializable).
"""

import asyncio
from typing import List, Callable
from pydantic import BaseModel

from reson.reson import agentic, Runtime


class SearchResult(BaseModel):
    """A search result with title and url."""
    title: str
    url: str
    snippet: str


class ResearchReport(BaseModel):
    """A research report with findings and sources."""
    topic: str
    summary: str
    key_findings: List[str]
    sources: List[SearchResult]


async def search_web(query: str) -> List[SearchResult]:
    """Mock web search function."""
    # In real usage, this would search the web
    return [
        SearchResult(
            title=f"Result for {query}",
            url=f"https://example.com/{query.replace(' ', '-')}",
            snippet=f"Information about {query}..."
        )
    ]


async def analyze_text(text: str, focus: str = "general") -> str:
    """Mock text analysis function."""
    return f"Analysis of '{text}' with focus on {focus}: Key insights found."


@agentic(model="openrouter:openai/gpt-4o")
async def research_agent(
    topic: str,
    search_web: Callable,
    analyze_text: Callable,
    runtime: Runtime
) -> ResearchReport:
    """
    Research a topic using web search and text analysis tools.
    
    The LLM can call:
    - search_web(query: str) to search for information
    - analyze_text(text: str, focus: str) to analyze findings

    To call a tool you simply respond with one of the return types.
    
    {{return_type}}
    """
    # The LLM will receive a Union[SearchWebTool, AnalyzeTextTool, ResearchReport]
    # as its output type, allowing it to choose which tool to call or return the final report
    
    result = await runtime.run()
    
    # Handle tool calls in a loop
    context = []
    max_iterations = 5
    
    for i in range(max_iterations):
        if runtime.is_tool_call(result):
            tool_name = runtime.get_tool_name(result)
            print(f"LLM called tool: {tool_name}")
            
            # Execute the tool
            tool_output = await runtime.execute_tool(result)
            print(f"Tool returned: {tool_output}")
            
            # Add to context
            context.append(f"{tool_name} returned: {tool_output}")
            
            # Continue with updated context
            prompt = f"""
            Research topic: {topic}
            
            Previous actions:
            {chr(10).join(context)}
            
            Continue researching or provide final ResearchReport.
            """
            
            result = await runtime.run(prompt=prompt)
        else:
            # Got the final report
            return result
    
    # If we hit max iterations, ask for final report
    final_prompt = f"""
    Research topic: {topic}
    
    Previous actions:
    {chr(10).join(context)}
    
    Please provide the final ResearchReport now.
    """
    
    return await runtime.run(prompt=final_prompt, output_type=ResearchReport)


async def main():
    """Run the research agent example."""
    print("Starting research agent with tool types...")
    
    # Research a topic
    report = await research_agent(
        topic="Recent advances in quantum computing",
        search_web=search_web,
        analyze_text=analyze_text
    )

    print(f"Report: {report}")
    
    print(f"\nFinal Research Report:")
    print(f"Topic: {report.topic}")
    print(f"Summary: {report.summary}")
    print(f"Key Findings:")
    for finding in report.key_findings:
        print(f"  - {finding}")
    print(f"Sources: {len(report.sources)} sources found")


if __name__ == "__main__":
    asyncio.run(main())
