import asyncio
from typing import List, Dict
from pydantic import BaseModel
from reson.reson import agentic, Runtime # Assuming reson.reson is accessible

# 1. Define a Pydantic model for the expected output
class AnalysisResult(BaseModel):
    summary: str
    keywords: List[str]
    sentiment: str # Added another field for a slightly more complex output

# 2. Create an agentic function with Jinja2 in its docstring
@agentic(model="openrouter:anthropic/claude-sonnet-4") # User might need OPENROUTER_KEY
async def analyze_text_data_with_jinja(
    user_input: str, 
    item_list: List[str], 
    config_param: Dict[str, str],
    runtime: Runtime
) -> AnalysisResult:
    """
    Analyze the provided user input: {{ user_input }}

    Consider the following items carefully:
    {% for item in item_list %}
    - Item {{ loop.index }}: {{ item }}
    {% endfor %}

    Current processing configuration: {{ config_param | json }}

    Based on all the above information, please provide a concise summary, 
    a list of relevant keywords, and the overall sentiment of the user input.
    
    {{return_type}}
    """
    # runtime.run() will use the rendered docstring as its prompt.
    # The TypeParser's enhance_prompt will handle the {{return_type}} placeholder.
    res = await runtime.run() 

    return res

# 3. Main function to run the test
async def main():
    sample_input = "The new product launch was a massive success, exceeding all expectations!"
    sample_items = ["positive feedback", "high sales", "good reviews"]
    sample_config = {"mode": "thorough_analysis", "language_preference": "en-US"}

    print(f"--- Calling analyze_text_data_with_jinja with: ---")
    print(f"User Input: '{sample_input}'")
    print(f"Items: {sample_items}")
    print(f"Config: {sample_config}")
    print(f"-------------------------------------------------")

    try:
        result = await analyze_text_data_with_jinja(
            user_input=sample_input, 
            item_list=sample_items,
            config_param=sample_config
        )
        print("\n--- Analysis Result: ---")
        print(f"Summary: {result.summary}")
        print(f"Keywords: {result.keywords}")
        print(f"Sentiment: {result.sentiment}")
        print(f"------------------------")
        
        print("\nTest PASSED if the output above is structured and makes sense based on the input.")
        print("This indicates that the Jinja2 templating in the docstring worked correctly.")

    except Exception as e:
        print(f"\n--- Error during test: ---")
        print(f"An error occurred: {type(e).__name__} - {e}")
        print(f"--------------------------")
        print("Troubleshooting tips:")
        print("1. Ensure the 'reson' library is correctly installed and accessible in your PYTHONPATH.")
        print("2. Make sure you have set the OPENROUTER_KEY environment variable if using OpenRouter.")
        print("3. Check your internet connection and that the model 'openrouter:perplexity/pplx-7b-online' is available.")
        print("4. Review the console for any specific error messages from the LLM or parsing stages.")

if __name__ == "__main__":
    asyncio.run(main())
