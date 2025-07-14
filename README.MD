# Minizica GenAI AgentOS

This is a custom AI-powered multi-agent automation platform designed to help developers and managers streamline their daily workflows. We built this solution because modern software teams waste valuable time switching between tools like Jira, GitHub, Confluence, Notion, and email—often repeating manual, error-prone tasks just to keep projects moving. To address this, we developed a system where a master agent can interpret a single natural language instruction and intelligently coordinate a suite of specialized agents to handle tasks like code analysis, code changes, documentation updates, ticket management, and communication. These agents are deeply integrated with popular developer tools via APIs and an agent orchestration layer, allowing them to work together seamlessly and automate even the most complex workflows—from updating a Jira ticket and pushing code to GitHub, to syncing documentation and sending email updates. The result is a highly flexible, extensible platform that saves time, reduces errors, keeps teams in sync, and empowers both developers and managers to focus on high-value work, rather than on repetitive overhead.

This repository extends and enhances the original [GenAI AgentOS](https://github.com/genai-works-org/genai-agentos) infrastructure to provide a rich platform for running, customizing, and orchestrating GenAI agents.

> **Note:**
> This project is **built on top of the original [genai-agentos](https://github.com/genai-works-org/genai-agentos)**.
> Please refer to [README-genai-agentos.md](README-genai-agentos.md) for full documentation, system architecture, prerequisites, and all upstream instructions.

---

## What's Included

This repo **inherits all features** of the upstream GenAI AgentOS:

* Full backend, frontend, router, PostgreSQL, Redis, Celery, and more.
* CLI and agent templates for rapid prototyping.
* Multi-agent orchestration, including Master Agents, A2A, and MCP server integration.

***Additionally, this repo adds custom agents and local enhancements for the Minizica project.***

---

## Quick Start

### 1. **Start Main Infrastructure (Docker)**

```bash
docker compose up
```

This will bring up all required services (backend, frontend, router, database, etc.) as per the standard AgentOS architecture.

---

### 2. **Start GenAI Agents**

Each agent runs independently. Open a new terminal for each agent you want to run, and execute the following commands **from the repo root**:

#### Analyse Code Agent

```bash
cd cli/agents/analyse_code_agent
uv run python analyse_code_agent.py
```

#### Coding Agent

```bash
cd cli/agents/coding_agent
uv run python coding_agent.py
```

#### Email Agent

```bash
cd cli/agents/email_agent
uv run python email_agent.py
```

#### GitHub Agent

```bash
cd cli/agents/github_agent
uv run python github_agent.py
```

#### Prompt Agent

```bash
cd cli/agents/prompt_agent
uv run python prompt_agent.py
```

#### Resource Finder Agent

```bash
cd cli/agents/resource_finder_agent
uv run python resource_finder_agent.py
```

---

### 3. **Start A2A Agents**

#### Speaking Agent

```bash
cd a2a_tts
uv run .
```

---

### 4. **Start MCP Gateway Servers via Docker**

> **MCP servers should be set up as Docker containers.**
> Download images from the [Docker MCP Catalog](https://hub.docker.com/search?q=mcp&type=image) and start the servers using the following commands:

#### Atlassian MCP Server

```bash
docker mcp gateway run --port 8899 --transport streaming --servers atlassian
```

#### Notion MCP Server

```bash
docker mcp gateway run --port 8888 --transport streaming --servers notion
```

#### Sequential Thinking MCP Server

```bash
docker mcp gateway run --port 8877 --transport streaming --servers sequentialthinking
```

#### DuckDuckGo MCP Server

```bash
docker mcp gateway run --port 8866 --transport streaming --servers duckduckgo
```

---

## Documentation

* **For upstream architecture, configuration, and full agent types:**
  Refer to [README-genai-agentos.md](README-genai-agentos.md)
* **For component-specific details:**
  See the respective READMEs:

  * [`cli/README.md`](cli/README.md)
  * [`backend/README.md`](backend/README.md)
  * [`master-agent/README.md`](master-agent/README.md)
  * [`router/README.md`](router/README.md)
  * [`frontend/README.md`](frontend/README.md)