"""
Example demonstrating the PostgreSQL JSONB store with cleanup.

This example shows how to use reson with a PostgreSQL backend for storage,
leveraging PostgreSQL's JSONB capabilities for document storage.
"""

import asyncio
from typing import List
from pydantic import BaseModel

from reson.reson import agentic, Runtime
from reson.stores import PostgresStoreConfig


class Person(BaseModel):
    """A person with a name, age, and optional list of skills."""
    name: str
    age: int
    skills: List[str] = []


@agentic(
    model="openrouter:openai/gpt-4o",
    store_cfg=PostgresStoreConfig(
        dsn="postgresql://postgres:postgres@localhost:5432/quarkus",
        table="agent_context",
        column="data"
    )
)
async def extract_and_store_people(text: str, runtime: Runtime) -> List[Person]:
    """
    Extract people information from the given text and store results in context.
    
    We'll store both the original input and extracted results in our PostgreSQL
    context store, demonstrating how to persist data between calls.
    
    Return a list of Person objects with names, ages, and skills.
    """
    # Store the original input in context
    await runtime.context.set("last_input", text)
    
    # Extract people from the text
    people = await runtime.run(prompt=f"Extract people from: {text}")
    
    # Store the extracted people in context
    await runtime.context.set("people_history", people)
    
    return people


@agentic(
    model="openrouter:openai/gpt-4o",
    store_cfg=PostgresStoreConfig(
        dsn="postgresql://postgres:postgres@localhost:5432/quarkus",
        table="agent_context",
        column="data"
    )
)
async def retrieve_stored_context(runtime: Runtime) -> dict:
    """
    Retrieve the stored context from previous runs.
    
    This demonstrates that the data is actually persisted in PostgreSQL
    between function calls.
    """
    # Get the stored context
    last_input = await runtime.context.get("last_input", "No previous input")
    people_history = await runtime.context.get("people_history", [])
    
    # Every agentic function must call runtime.run() or runtime.run_stream() at least once
    # Making a minimal call here to satisfy this requirement
    await runtime.run(prompt="Acknowledge that you've retrieved the context data successfully.")
    
    return {
        "last_input": last_input,
        "people_history": people_history
    }


async def cleanup_postgres_table():
    """
    Clean up by dropping the table used in this example.
    
    In a real application, you might not want to do this, but for an example
    it's good to clean up after ourselves.
    """
    from reson.data.postgres.manager import DatabaseManager
    
    # Connect to the database
    db = DatabaseManager("postgresql://postgres:postgres@localhost:5432/quarkus")
    
    # Drop the table if it exists
    db.execute_query("DROP TABLE IF EXISTS agent_context")
    
    print("PostgreSQL table 'agent_context' dropped.")


async def main():
    """Run the example."""
    print("\n=== PostgreSQL JSONB Store Example ===\n")
    
    # First, extract people from a text
    text = "John is 35 and knows Python and JavaScript. Maria is 28 and is skilled in data science and machine learning."
    print(f"Input text: {text}\n")
    
    # The @agentic decorator transforms the function
    # When we call the function by name, the decorator handles creating the runtime
    
    # Extract and store people
    print("Extracting people...")
    # The decorator automatically injects the runtime parameter
    people = await extract_and_store_people(text)
    
    print("\nExtracted people:")
    for person in people:
        print(f"- {person.name}, {person.age}: {', '.join(person.skills)}")
    
    # Retrieve stored context
    print("\nRetrieving stored context...")
    # The decorator automatically injects the runtime parameter
    context = await retrieve_stored_context()
    
    print("\nStored context:")
    print(f"Last input: {context['last_input']}")
    print("People history:")
    for person in context['people_history']:
        print(f"- {person.name}, {person.age}: {', '.join(person.skills)}")
    
    # Clean up
    print("\nCleaning up...")
    await cleanup_postgres_table()
    
    print("\nExample completed!")


if __name__ == "__main__":
    asyncio.run(main())
