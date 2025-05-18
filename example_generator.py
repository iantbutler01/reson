"""
Example demonstrating the generator yield semantics for agentic functions.

This example shows how to create an agentic function that yields values,
which can be consumed with an async for loop.
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, List
from pydantic import BaseModel

from reson.reson import agentic, Runtime


class DataItem(BaseModel):
    """A single data item with an ID and value."""
    id: int
    value: str
    processed: bool = False


@agentic(model="openrouter:openai/gpt-4o")
async def process_data_stream(data: List[str], runtime: Runtime) -> AsyncGenerator[DataItem, None]:
    """
    Process each item in the data list one by one and yield the results.
    
    For each item, create a DataItem with:
    - An ID (sequential number)
    - A value (the processed version of the input)
    - processed flag set to True
    
    Yield each item as it's processed.
    """
    for i, item in enumerate(data):
        prompt = f"""
        Process this data item: "{item}"
        
        Return a more informative version with additional context or details.
        """
        
        result = await runtime.run(prompt=prompt, output_type=str)
        
        # Create a DataItem and yield it
        data_item = DataItem(
            id=i + 1,
            value=result,
            processed=True
        )
        
        yield data_item


@agentic(model="openrouter:openai/gpt-4o")
async def process_data_with_status(data: List[str], runtime: Runtime) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process items and yield status updates along with results.
    
    This demonstrates how an agentic function can yield both status updates
    and actual results as the processing happens.
    """
    total = len(data)
    
    # First yield the initial status
    yield {
        "type": "status",
        "message": f"Starting to process {total} items",
        "progress": 0,
        "total": total
    }
    
    # Process each item
    for i, item in enumerate(data):
        # Yield a progress update
        yield {
            "type": "status",
            "message": f"Processing item {i+1}/{total}",
            "progress": i,
            "total": total
        }
        
        # Process the item
        prompt = f"Summarize this text in one sentence: {item}"
        result = await runtime.run(prompt=prompt, output_type=str)
        
        # Yield the actual result
        yield {
            "type": "result",
            "item_id": i,
            "original": item,
            "summary": result
        }
    
    # Final status update
    yield {
        "type": "status",
        "message": "Processing complete",
        "progress": total,
        "total": total
    }


async def main():
    """Run the example."""
    print("\n=== Example 1: Basic Generator ===")
    data = [
        "Climate change is affecting global weather patterns",
        "Artificial intelligence continues to advance rapidly",
        "Space exploration is entering a new commercial phase"
    ]
    
    print("Processing data items one by one:")
    async for item in process_data_stream(data):
        print(f"Item {item.id}: {item.value}")
        print(f"Processed: {item.processed}")
        print()
    
    print("\n=== Example 2: Generator with Status Updates ===")
    articles = [
        "The city council voted yesterday to approve the new budget for next year, which includes increased funding for public transportation and parks.",
        "Scientists have discovered a new species of deep-sea creature that can survive extreme pressure and temperature conditions."
    ]
    
    print("Processing with status updates:")
    async for update in process_data_with_status(articles):
        if update["type"] == "status":
            print(f"Status: {update['message']} ({update['progress']}/{update['total']})")
        else:
            print(f"Result for item {update['item_id']}:")
            print(f"Original: {update['original'][:50]}...")
            print(f"Summary: {update['summary']}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
