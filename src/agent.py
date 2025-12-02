from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext
from pathlib import Path

from .models import TestPlan
from .settings import settings


@dataclass
class AgentDeps:
    http_client: httpx.AsyncClient
    repo_path: Path | None = None
    fetched_openapi_spec: dict | None = None  # Stores {url: str, content: str}


system_prompt = """
You are a QA Automation Architect. Your goal is to create a robust, parallelizable test plan for a REST API.

You will be provided with:
1. An OpenAPI specification (via URL).
2. Access to the source code repository (optional).

Your task:
1. Analyze the OpenAPI spec to understand the available endpoints and data models.
2. If available, analyze the source code to understand business logic, constraints, and hidden dependencies.
3. Generate a dependency graph of test steps.
    - CRUD operations usually have dependencies (e.g., Create User -> Get User -> Update User -> Delete User).
    - Independent operations should not depend on each other to allow parallel execution.
    - Ensure the 'depends_on' field correctly reflects these relationships.
4. Output the plan as a structured JSON object.
"""

agent = Agent(
    model=f"openai:{settings.openai_model}",
    deps_type=AgentDeps,
    output_type=TestPlan,
    system_prompt=system_prompt,
)


@agent.tool
async def fetch_openapi_spec(ctx: RunContext[AgentDeps], url: str) -> str:
    """
    Fetch the OpenAPI specification from a given URL.
    """
    try:
        response = await ctx.deps.http_client.get(url)
        response.raise_for_status()
        content = response.text
        # Store the fetched spec in the state
        ctx.deps.fetched_openapi_spec = {"url": url, "content": content}
        return content
    except Exception as e:
        return f"Error fetching OpenAPI spec: {str(e)}"


@agent.tool
async def list_files(ctx: RunContext[AgentDeps], directory: str = ".") -> str:
    """
    List files in the repository to understand the project structure.
    Only works if a repo path was provided.
    """
    if not ctx.deps.repo_path:
        return "No repository path provided."

    target_path = (ctx.deps.repo_path / directory).resolve()

    # Security check to ensure we don't escape the repo path
    if not str(target_path).startswith(str(ctx.deps.repo_path.resolve())):
        return "Access denied: Cannot access files outside the repository."

    if not target_path.exists():
        return f"Directory not found: {directory}"

    try:
        files = [f.name for f in target_path.iterdir()]
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"


@agent.tool
async def read_file(ctx: RunContext[AgentDeps], filepath: str) -> str:
    """
    Read the content of a specific file in the repository.
    Only works if a repo path was provided.
    """
    if not ctx.deps.repo_path:
        return "No repository path provided."

    target_path = (ctx.deps.repo_path / filepath).resolve()

    # Security check
    if not str(target_path).startswith(str(ctx.deps.repo_path.resolve())):
        return "Access denied: Cannot access files outside the repository."

    if not target_path.exists():
        return f"File not found: {filepath}"

    try:
        return target_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"
