import asyncio
from typing import Annotated
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

import psycopg2
import openai
import re
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = dict(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZjBjODY2Mi1kOTZmLTRhYmQtODM1NC0wZTNiYzZjMmZmZjYiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImI5OWQ4YTNmLWFjNTItNGRmNC05YjExLTYxNzljZDQ4Y2MxNSJ9.wwHaP0uNMQiEP7su-gL0GkXGZZ-5LnfrnEfJOVRCOCI"  # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)

# --------- LLM UTILS ---------
def extract_keywords_from_query(query):
    prompt = f"Extract only the most relevant search keywords (comma-separated, no extra text) from this question: '{query}'\nKeywords:"
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract only the most relevant keywords, comma-separated, no extra text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=32,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    keywords = []
    for k in raw.split(","):
        k = k.strip()
        if k:
            # Split on whitespace, add all words
            words = k.split()
            keywords.extend([w for w in words if w])

    # ---- FORCE INCLUDE domain keywords if present in query ----
    boost_terms = ["code", "documentation", "document", "docs", "file", "files"]
    query_lower = query.lower()
    for term in boost_terms:
        if term in query_lower and term not in [k.lower() for k in keywords]:
            keywords.append(term)
    return keywords

def extract_number_from_llm_response(resp_content, default=1):
    # Extract first integer from string; if none, return default
    match = re.search(r'\d+', resp_content)
    if match:
        return int(match.group())
    else:
        return default
    
def llm_select_best_resource(user_question, candidate_entries):
    prompt = f'A user asked: "{user_question}"\nHere are possible knowledge domains:\n'
    for idx, entry in enumerate(candidate_entries, 1):
        prompt += f"{idx}. {entry['platform_name']} / {entry['main_project']}: {entry['summary']}\n"
    prompt += "\nBased on the user’s question, which one is the most relevant? Respond with the number only."
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Pick ONLY the number of the best matching resource based on the user's question. Respond with just the number."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4,
        temperature=0
    )
    resp_content = response.choices[0].message.content.strip()
    choice = extract_number_from_llm_response(resp_content, default=1)
    # Always return at least one result (fallback to first if invalid or out of range)
    if choice < 1 or choice > len(candidate_entries):
        choice = 1
    return candidate_entries[choice - 1]


def llm_select_best_detail(user_query, detail_entries):
    prompt = f'User asked: "{user_query}"\nHere are possible resources:\n'
    for idx, entry in enumerate(detail_entries, 1):
        prompt += f"{idx}. Title: {entry.get('title','')}\n   Summary: {entry.get('summary','')}\n   How to access: {entry.get('how_to_access','')}\n"
    prompt += "\nBased on the user’s question, which resource is most relevant? Respond with the number only."
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Pick ONLY the number of the most relevant resource based on the user's question. Respond with just the number."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4,
        temperature=0
    )
    choice = int(response.choices[0].message.content.strip())
    return detail_entries[choice - 1]

# --------- SQL Helper ---------
def build_or_query(fields, keywords):
    # Returns: (query_str, params)
    clauses = []
    params = []
    for kw in keywords:
        for f in fields:
            clauses.append(f"{f} ILIKE %s")
            params.append(f"%{kw}%")
    return " OR ".join(clauses), params

def score_row(row, keywords, field_map):
    score = 0
    boost_keywords = {"code", "documentation", "document", "docs", "file", "files"}
    boost_weight = 1000.0  # You can adjust this
    for kw in keywords:
        kw_lower = kw.lower()
        for field, weight in field_map.items():
            value = row.get(field)
            # Handle arrays (tags, keywords) as well as strings
            if isinstance(value, list):
                for v in value:
                    if kw_lower in str(v).lower():
                        score += weight
                        if kw_lower in boost_keywords:
                            score += boost_weight
            elif value and kw_lower in str(value).lower():
                score += weight
                if kw_lower in boost_keywords:
                    score += boost_weight
    return score

# --------- MAIN AGENT LOGIC ---------
def get_resource_and_instructions(user_query):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # 1. Extract keywords (LLM)
    keywords = extract_keywords_from_query(user_query)
    print(f"Extracted keywords: {keywords}")
    
    # 2. Search resource_catalog for possible matches using all keywords
    where_str, params = build_or_query(
        ['summary', 'main_project', 'platform_name'],
        keywords
    )
    cur.execute(f"""
        SELECT platform_name, main_project, summary, associated_table
        FROM resource_catalog
        WHERE {where_str}
    """, tuple(params))
    rows = cur.fetchall()
    if not rows:
        cur.close(); conn.close()
        return {"result": None, "message": "No relevant resource found in catalog."}

    candidates = [
        {
            "platform_name": r[0],
            "main_project": r[1],
            "summary": r[2],
            "associated_table": r[3]
        }
        for r in rows
    ]
    # ----- Weighted Scoring for Main Table -----
    main_field_map = {"platform_name": 2.0, "main_project": 2.0, "summary": 1.0}
    boost_keywords = {"code", "documentation", "document", "docs", "file", "files"}
    for c in candidates:
        c['score'] = score_row(c, keywords, main_field_map)
        # Extra boost if code/doc keywords in summary or main_project
        text = (c['summary'] + " " + c['main_project']).lower()
        if any(bk in text for bk in boost_keywords):
            if any(kw in boost_keywords for kw in [k.lower() for k in keywords]):
                c['score'] += 2.0  # Adjust this boost as needed
    candidates = [c for c in candidates if c['score'] > 0]
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    if not candidates:
        cur.close(); conn.close()
        return {"result": None, "message": "No relevant resource found in catalog after scoring."}
    # 3. If multiple candidates, let LLM choose (optional, can also pick top N)
    best = candidates[0]

    table_name = best["associated_table"]
    platform = best["platform_name"]
    project = best["main_project"]

    # 4. Now search in the associated table using all keywords
    results = []
    if table_name == "code_files":
        where_str, params = build_or_query(
            ['summary', 'file_name', 'access_instructions'],
            keywords
        )
        # Add keywords array field
        where_str += " OR " + " OR ".join(["array_to_string(keywords,',') ILIKE %s" for _ in keywords])
        params += [f"%{kw}%" for kw in keywords]
        cur.execute(f"""
            SELECT access_instructions, file_name, summary, keywords
            FROM code_files
            WHERE {where_str}
        """, tuple(params))
        details = cur.fetchall()
        for d in details:
            results.append({
                "how_to_access": d[0],
                "title": d[1],
                "summary": d[2],
                "keywords": d[3]
            })
        field_map = {"title": 2.0, "keywords": 1.5, "how_to_access": 1.5, "summary": 1.0}
    elif table_name == "doc_index":
        where_str, params = build_or_query(
            ['summary', 'title', 'access_instructions'],
            keywords
        )
        where_str += " OR " + " OR ".join(["array_to_string(tags,',') ILIKE %s" for _ in keywords])
        params += [f"%{kw}%" for kw in keywords]
        cur.execute(f"""
            SELECT access_instructions, title, summary, url, tags
            FROM doc_index
            WHERE {where_str}
        """, tuple(params))
        details = cur.fetchall()
        for d in details:
            results.append({
                "how_to_access": d[0],
                "title": d[1],
                "summary": d[2],
                "url": d[3],
                "keywords": d[4]
            })
        field_map = {"title": 2.0, "keywords": 1.5, "how_to_access": 1.5, "summary": 1.0}
    elif table_name == "notion_files":
        where_str, params = build_or_query(
            ['summary', 'title', 'access_instructions'],
            keywords
        )
        where_str += " OR " + " OR ".join(["array_to_string(keywords,',') ILIKE %s" for _ in keywords])
        params += [f"%{kw}%" for kw in keywords]
        cur.execute(f"""
            SELECT access_instructions, title, summary, url, keywords
            FROM notion_files
            WHERE {where_str}
        """, tuple(params))
        details = cur.fetchall()
        for d in details:
            results.append({
                "how_to_access": d[0],
                "title": d[1],
                "summary": d[2],
                "url": d[3],
                "keywords": d[4]
            })
        field_map = {"title": 2.0, "keywords": 1.5, "how_to_access": 1.5, "summary": 1.0}
    else:
        results = []

    cur.close(); conn.close()

    # 5. Weighted scoring for detail table (with boost)
    for r in results:
        r['score'] = score_row(r, keywords, field_map)
    results = [r for r in results if r['score'] > 0]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    if results:
        # Use LLM to pick the best from the top 3
        top_candidates = results[:3]
        best_entry = llm_select_best_detail(user_query, top_candidates)
        output = {
            "platform": platform,
            "project": project,
            "source_table": table_name,
            "navigation": {
                "title": best_entry.get("title"),
                "how_to_access": best_entry.get("how_to_access"),
                "url": best_entry.get("url"),
                "keywords": best_entry.get("keywords"),
            },
            "summary": best_entry.get("summary"),
        }
        return {"result": output, "message": "Success"}
    else:
        return {
            "result": {
                "platform": platform,
                "project": project,
                "source_table": table_name,
                "navigation": None,
                "summary": f"Relevant area found: {platform} / {project}, but no specific entry matched your keywords."
            },
            "message": "No specific detailed match found."
        }

# ---- AGENT BINDING ----
@session.bind(
    name="resource_finder_agent",
    description="Finds the most relevant code or document based on a user’s question and tells exactly how to access it in a machine-readable format for further navigation."
)
async def resource_finder_agent(
    agent_context: GenAIContext,
    user_query: Annotated[
        str,
        "The user's question or request for a resource. The agent will find the most relevant code or document and explain how to access it.",
    ],
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, get_resource_and_instructions, user_query
    )
    return result

async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
