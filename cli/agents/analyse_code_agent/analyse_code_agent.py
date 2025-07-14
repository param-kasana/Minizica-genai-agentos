import asyncio
from typing import Annotated, Dict
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
import os
import openai
from dotenv import load_dotenv

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4ZWFkN2ZhMi00ZGQ1LTQ0MWQtYjk3NC02NzczM2RmN2VjMzMiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImI5OWQ4YTNmLWFjNTItNGRmNC05YjExLTYxNzljZDQ4Y2MxNSJ9.ZXSDZ6TxYus9DOoDg674WALYw5VCR42WZ-Q-94R3QnU" 
session = GenAISession(jwt_token=AGENT_JWT)

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

@session.bind(
    name="analyse_code_agent",
    description=(
        "Analyzes provided code files to find and summarize those most relevant to a given feature description. "
        "Returns focused summaries and targeted code snippets. "
        "Requires both a feature description (string) and a file_contents argument (dictionary mapping filenames to code as strings)."
    ),
)
async def analyse_code_agent(
    agent_context: GenAIContext,
    feature_description: Annotated[
        str,
        "A short description of the feature to analyze for. Example: 'Navigation bar with sticky header and responsive menu.'"
    ],
    file_contents: Annotated[
        Dict[str, str],
        "A dictionary where keys are filenames (str) and values are the full code contents of those files (str). Example: { 'app.py': '...code...', 'templates/skills.html': '...code...' }"
    ],
):
    """
    Arguments Required:
        feature_description (str): A short description of the feature you want to implement/analyze.
        file_contents (dict): Dictionary of { filename: code_as_string } for all files to be analyzed.

    Example:
        feature_description = "Add a navigation bar"
        file_contents = {
            "navigation_bar.js": "...",
            "App.js": "...",
            "templates/skills.html": "..."
        }
    """

    # Argument checking
    missing = []
    if not feature_description or not isinstance(feature_description, str):
        missing.append("feature_description (string)")
    if not file_contents or not isinstance(file_contents, dict) or not any(isinstance(v, str) for v in file_contents.values()):
        missing.append("file_contents (dict of {filename: code_as_string})")
    if missing:
        return {
            "error": (
                f"Missing or invalid arguments: {', '.join(missing)}.\n\n"
                "Please provide:\n"
                " - feature_description (str): A short description of the feature to analyze for.\n"
                " - file_contents (dict): A dictionary of {filename: code} for the files to be analyzed.\n"
                "Example:\n"
                "{\n"
                '  "feature_description": "Create a responsive navigation bar",\n'
                '  "file_contents": {\n'
                '    "app.py": "def app(): ...",\n'
                '    "templates/skills.html": "<div> ... </div>"\n'
                "  }\n"
                "}"
            )
        }

    keywords = set(feature_description.lower().split())
    relevant_files = set()
    for fname, content in file_contents.items():
        fname_lower = fname.lower()
        content_lower = content.lower()
        if any(word in fname_lower for word in keywords):
            relevant_files.add(fname)
        elif any(word in content_lower for word in keywords):
            relevant_files.add(fname)

    results = []
    if OPENAI_KEY and relevant_files:
        client = openai.OpenAI(api_key=OPENAI_KEY)
        for fname in relevant_files:
            code = file_contents.get(fname, "")[:5000]  # limit for LLM
            extract_prompt = (
                f"You are a software engineer. Given the feature description and the following code file, "
                f"extract ONLY the classes, functions, or code sections most relevant for implementing the feature. "
                f"Also, provide a 1-2 sentence summary of what the file is responsible for.\n\n"
                f"Feature Description:\n{feature_description}\n\n"
                f"File: {fname}\n"
                f"Code:\n{code}\n\n"
                f"---\n"
                f"Return your answer as:\n"
                f"Summary: <summary here>\n"
                f"Relevant Code:\n<relevant code here>"
            )
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior software engineer."},
                    {"role": "user", "content": extract_prompt}
                ],
                max_tokens=1024,
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            summary, relevant_code = "", ""
            if "Relevant Code:" in content:
                parts = content.split("Relevant Code:", 1)
                summary = parts[0].replace("Summary:", "").strip()
                relevant_code = parts[1].strip()
            else:
                summary = content
            results.append({
                "filename": fname,
                "summary": summary,
                "relevant_code": relevant_code
            })
    else:
        # Fallback: just summary and (if file is small) full code
        for fname in relevant_files:
            code = file_contents.get(fname, "")
            short_code = code if len(code) < 2000 else "(File too large, review manually)"
            results.append({
                "filename": fname,
                "summary": f"Likely related to: {fname.replace('.js','').replace('_',' ').title()}",
                "relevant_code": short_code
            })

    return {"relevant_files": results}


async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
