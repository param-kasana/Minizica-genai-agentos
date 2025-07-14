import asyncio
import os
from typing import Annotated, Optional
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from pydantic import BaseModel
import openai
from dotenv import load_dotenv

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI5MmMwZjgwOS1hZDYyLTQ3ZjEtOGRjOC0zMjY2ZjhhNWMwNWYiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImI5OWQ4YTNmLWFjNTItNGRmNC05YjExLTYxNzljZDQ4Y2MxNSJ9.7NViIohFih2Z0dh9Z-mgxtQ6w2c0THwrx-6Ep-3NIhE" # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

class CodeRequest(BaseModel):
    mode: str
    description: str
    language: str
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    existing_code: Optional[str] = None
    additional_context: Optional[str] = None

def generate_detailed_prompt(request: CodeRequest) -> str:
    meta_prompt = f"""
You are an expert prompt engineer designing prompts for an advanced LLM coding assistant.

Given this structured coding request, generate a highly detailed, best-practice natural language prompt that will instruct the coding LLM to:
- (1) Write clean, idiomatic code in the requested language
- (2) Include comprehensive docstrings and in-line comments for maintainability
- (3) Handle all relevant edge cases and input validation
- (4) For 'fix' mode: Explain what the bug was and how the fix solves it in a summary comment at the top
- (5) For 'create' mode: Use best practices for naming, modularity, and performance. Generate a robust function, class, or module as needed.
- (6) For 'refactor' mode: Modernize and improve code style, performance, or readability as appropriate, using the latest language features
- (7) Follow these instructions precisely, and return only the code block(s), with no additional explanation or non-code output.

Request Details:
Mode: {request.mode}
Description: {request.description}
Language: {request.language}
File Path: {request.file_path}
Function/Class Name: {request.function_name}
Existing Code: {request.existing_code}
Additional Context: {request.additional_context}

Meta-instructions: If any request field is missing, use your best judgement. ALWAYS err on the side of detailed, production-quality code. For code fixes, ensure the bug is actually solved. For code generation, generate code as if it is going into a large-scale production codebase.
DO NOT WRITE THE CODE YOURSELF, JUST GENERATE THE PROMPT.
"""
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are the world's best prompt engineer for coding LLMs."},
            {"role": "user", "content": meta_prompt}
        ],
        max_tokens=2048
    )
    detailed_prompt = response.choices[0].message.content.strip()
    return detailed_prompt

@session.bind(
    name="prompt_agent",
    description="Helps in generating Prompts. Should be used for creating detailed prompts for coding tasks.",
)
async def prompt_agent(
    agent_context: GenAIContext,
    mode: Annotated[str, "Task mode: create, fix, or refactor"],
    description: Annotated[str, "Description of the coding task"],
    language: Annotated[str, "Programming language"],
    file_path: Annotated[Optional[str], "Target file path"] = None,
    function_name: Annotated[Optional[str], "Function or class name"] = None,
    existing_code: Annotated[Optional[str], "Existing code to modify or fix"] = None,
    additional_context: Annotated[Optional[str], "Any additional context"] = None,
):
    """Generates a detailed prompt for coding LLMs based on the provided request."""
    req = CodeRequest(
        mode=mode,
        description=description,
        language=language,
        file_path=file_path,
        function_name=function_name,
        existing_code=existing_code,
        additional_context=additional_context,
    )
    prompt = generate_detailed_prompt(req)
    return prompt

async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
