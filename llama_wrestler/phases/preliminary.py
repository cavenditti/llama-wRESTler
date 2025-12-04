from dataclasses import dataclass
from pathlib import Path
import httpx
import json
import logging

from llama_wrestler.agent import agent, AgentDeps
from llama_wrestler.models import APIPlan, APIStep
from llama_wrestler.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class PreliminaryResult:
    """Result from the preliminary phase."""

    test_plan: APIPlan
    openapi_spec: dict | None
    openapi_url: str


def extract_endpoints_from_spec(openapi_spec: dict) -> set[tuple[str, str]]:
    """
    Extract all endpoints (method, path) from an OpenAPI specification.

    Args:
        openapi_spec: Parsed OpenAPI specification dict

    Returns:
        Set of (method, path) tuples representing all endpoints
    """
    endpoints = set()
    paths = openapi_spec.get("paths", {})

    for path, path_item in paths.items():
        # OpenAPI methods are lowercase in the spec
        http_methods = ["get", "post", "put", "delete", "patch", "options", "head"]
        for method in http_methods:
            if method in path_item:
                endpoints.add((method.upper(), path))

    return endpoints


def get_covered_endpoints(plan: APIPlan) -> set[tuple[str, str]]:
    """
    Extract all endpoints covered by the test plan.

    Args:
        plan: The API test plan

    Returns:
        Set of (method, endpoint) tuples covered by the plan
    """
    return {(step.method, step.endpoint) for step in plan.steps}


def find_missing_endpoints(openapi_spec: dict, plan: APIPlan) -> set[tuple[str, str]]:
    """
    Find endpoints in the OpenAPI spec that are not covered by the test plan.

    Args:
        openapi_spec: Parsed OpenAPI specification dict
        plan: The current API test plan

    Returns:
        Set of (method, path) tuples that are missing from the plan
    """
    spec_endpoints = extract_endpoints_from_spec(openapi_spec)
    covered_endpoints = get_covered_endpoints(plan)
    return spec_endpoints - covered_endpoints


def merge_api_plans(base_plan: APIPlan, additional_plan: APIPlan) -> APIPlan:
    """
    Merge additional steps into the base plan deterministically.

    Steps from additional_plan are appended to base_plan if they cover
    new endpoints. Duplicate endpoint coverage is avoided.

    Args:
        base_plan: The base test plan
        additional_plan: Plan with additional steps to merge

    Returns:
        Merged APIPlan with all unique steps
    """
    covered = get_covered_endpoints(base_plan)
    existing_ids = {step.id for step in base_plan.steps}

    new_steps: list[APIStep] = []
    for step in additional_plan.steps:
        endpoint_key = (step.method, step.endpoint)
        if endpoint_key not in covered:
            # Ensure unique step ID
            new_id = step.id
            counter = 1
            while new_id in existing_ids:
                new_id = f"{step.id}_{counter}"
                counter += 1

            # Create step with potentially updated ID
            if new_id != step.id:
                step = step.model_copy(update={"id": new_id})

            new_steps.append(step)
            covered.add(endpoint_key)
            existing_ids.add(new_id)

    return APIPlan(
        summary=base_plan.summary,
        base_url=base_plan.base_url,
        steps=base_plan.steps + new_steps,
    )


async def run_preliminary_phase(
    openapi_url: str,
    repo_path: Path | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> PreliminaryResult:
    """
    Run the preliminary phase: analyze OpenAPI spec and generate test plan.

    This function ensures full endpoint coverage by:
    1. Generating an initial test plan
    2. Checking against the OpenAPI spec for missing endpoints
    3. Running additional passes (up to settings.max_coverage_passes) to cover missing endpoints
    4. Merging results deterministically

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
        current_plan = result.output

        # Extract OpenAPI spec content if it was fetched
        openapi_spec = None
        if deps.fetched_openapi_spec:
            openapi_spec = json.loads(deps.fetched_openapi_spec["content"])

        # Check for missing endpoints and run additional passes if needed
        if openapi_spec:
            for pass_num in range(1, settings.max_coverage_passes + 1):
                missing = find_missing_endpoints(openapi_spec, current_plan)

                if not missing:
                    logger.info(
                        f"Full endpoint coverage achieved after {pass_num} pass(es)"
                    )
                    break

                logger.info(
                    f"Pass {pass_num}: Found {len(missing)} missing endpoints, "
                    f"running additional generation..."
                )

                # Format missing endpoints for the prompt
                missing_list = "\n".join(
                    f"  - {method} {path}" for method, path in sorted(missing)
                )

                # Create a new deps instance that already has the spec cached
                additional_deps = AgentDeps(
                    http_client=client,
                    repo_path=repo_path,
                    fetched_openapi_spec=deps.fetched_openapi_spec,
                )

                additional_prompt = f"""Create test steps for the following endpoints that are missing from the current test plan.
The API spec is at {openapi_url}.

Missing endpoints:
{missing_list}

Use the same base_url as before: {current_plan.base_url}

IMPORTANT:
- Only create steps for the missing endpoints listed above
- Consider dependencies with existing steps when setting depends_on
- Use appropriate auth_requirement based on the OpenAPI spec security definitions
- Set correct body_format based on the endpoint's requestBody content type
"""

                try:
                    additional_result = await agent.run(
                        additional_prompt, deps=additional_deps
                    )
                    current_plan = merge_api_plans(
                        current_plan, additional_result.output
                    )
                except Exception as e:
                    logger.warning(
                        f"Pass {pass_num} failed to generate additional steps: {e}"
                    )
                    break

            # Log final coverage status
            final_missing = find_missing_endpoints(openapi_spec, current_plan)
            if final_missing:
                logger.warning(
                    f"Could not achieve full coverage. Missing {len(final_missing)} endpoints: "
                    f"{sorted(final_missing)}"
                )
            else:
                logger.info(
                    f"Final plan covers all {len(extract_endpoints_from_spec(openapi_spec))} endpoints"
                )

        return PreliminaryResult(
            test_plan=current_plan,
            openapi_spec=openapi_spec,
            openapi_url=openapi_url,
        )

    if http_client:
        return await _run_with_client(http_client)
    else:
        async with httpx.AsyncClient() as client:
            return await _run_with_client(client)
