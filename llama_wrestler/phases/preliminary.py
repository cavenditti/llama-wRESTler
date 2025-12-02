from dataclasses import dataclass
from pathlib import Path
import httpx

from llama_wrestler.agent import agent, AgentDeps
from llama_wrestler.models import TestPlan


@dataclass
class PreliminaryResult:
    """Result from the preliminary phase."""

    test_plan: TestPlan
    openapi_spec: dict | None
    openapi_url: str


async def run_preliminary_phase(
    openapi_url: str,
    repo_path: Path | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> PreliminaryResult:
    """
    Run the preliminary phase: analyze OpenAPI spec and generate test plan.

    Args:
        openapi_url: URL to the OpenAPI specification
        repo_path: Optional path to local repository for analysis
        http_client: Optional HTTP client (will create one if not provided)

    Returns:
        PreliminaryResult containing the test plan and fetched OpenAPI spec
    """

    async def _run_with_client(client: httpx.AsyncClient) -> PreliminaryResult:
        deps = AgentDeps(http_client=client, repo_path=repo_path)

        prompt = f"Create a test plan for the API at {openapi_url}."
        if repo_path:
            prompt += " Use the provided tools to analyze the repository code for better context."

        result = await agent.run(prompt, deps=deps)

        # Extract OpenAPI spec content if it was fetched
        openapi_spec = None
        if deps.fetched_openapi_spec:
            import json

            openapi_spec = json.loads(deps.fetched_openapi_spec["content"])

        return PreliminaryResult(
            test_plan=result.output,
            openapi_spec=openapi_spec,
            openapi_url=openapi_url,
        )

    if http_client:
        return await _run_with_client(http_client)
    else:
        async with httpx.AsyncClient() as client:
            return await _run_with_client(client)
