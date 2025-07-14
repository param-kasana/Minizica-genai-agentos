import asyncio
import os
from typing import Annotated
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmZDZmMWExMC00ZjhmLTQ0YWUtOTVkZS00MWYxYTk2YzI2OTYiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImI5OWQ4YTNmLWFjNTItNGRmNC05YjExLTYxNzljZDQ4Y2MxNSJ9.T9gPOa9vPSzOQazgykAeKwE0Qc-0AwiT7LFnuDlj8Ak"
session = GenAISession(jwt_token=AGENT_JWT)


@session.bind(
    name="coding_agent",
    description="Helps in Coding. Should be used for fixing or generating code.",
)
async def coding_agent(
    agent_context: GenAIContext,
    prompt: Annotated[
        str,
        "The code or coding task you want help with. For bug fixes, paste the code and describe the issue. For new code, describe the requirements.",  # noqa: E501
    ],
):
    """Fixes or generates code using OpenAI GPT-4."""
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class coding assistant. "
                    "You will always output ONLY executable code—never extra explanation, "
                    "markdown, or chatty text. If a prompt asks for a fix, ensure the bug is solved. "
                    "For new code, ensure it is clean, idiomatic, and ready for production. "
                    "For all tasks, include docstrings and comments as instructed in the prompt."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2048
    )
    code = response.choices[0].message.content.strip()
    return code


async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
