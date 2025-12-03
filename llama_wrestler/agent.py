from dataclasses import dataclass
import httpx
from pydantic_ai import Agent, RunContext
from pathlib import Path

from llama_wrestler.models import APIPlan
from llama_wrestler.settings import settings


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
4. Identify and properly handle authentication:
    - Mark login/token endpoints as AUTH_PROVIDER
    - Make all protected endpoints depend on the authentication step
    - Mark public endpoints as NONE
5. Set the correct body_format for each endpoint:
    - Use "form_urlencoded" for OAuth2 token endpoints (/token, /login with OAuth2PasswordRequestForm)
    - Use "multipart" for file upload endpoints
    - Use "json" for standard API calls with JSON bodies
    - Use "none" for GET requests and DELETE without body
6. Output the plan as a structured JSON object.

AUTHENTICATION HANDLING:
- Look for OAuth2 security schemes in the OpenAPI spec
- OAuth2 password flow endpoints (requestBody with x-www-form-urlencoded) MUST use body_format="form_urlencoded"
- Set auth_token_path to the JSON path where the token appears (e.g., "access_token")
- All endpoints requiring authentication should:
  - Have auth_requirement="required"
  - Have depends_on including the auth step ID
  - Use the token via headers: {"Authorization": "Bearer {{auth_step_id.access_token}}"}

BODY FORMAT DETECTION:
- Check the "requestBody.content" in OpenAPI spec:
  - "application/json" -> body_format="json"
  - "application/x-www-form-urlencoded" -> body_format="form_urlencoded"
  - "multipart/form-data" -> body_format="multipart"
- If no requestBody -> body_format="none"

DEPENDENCY GRAPH RULES:
- Authentication endpoints should have NO dependencies (they come first)
- Protected CRUD operations should depend on auth + any data dependencies
- Example flow: auth_login -> create_item -> get_item -> update_item -> delete_item
"""

agent = Agent(
    model=settings.get_model(),
    deps_type=AgentDeps,
    output_type=APIPlan,
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
