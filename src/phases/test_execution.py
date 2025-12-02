from dataclasses import dataclass, field
from typing import Any
import httpx
import re
from pydantic import BaseModel, Field

from ..models import TestPlan, TestStep, BodyFormat, AuthRequirement
from .data_generation import GeneratedTestData, MockedPayload


class StepResult(BaseModel):
    """Result of executing a single test step."""

    step_id: str
    success: bool
    status_code: int | None = None
    expected_status: int
    response_body: Any = None
    error: str | None = None
    duration_ms: float = 0
    auth_token_extracted: str | None = None  # Token if this was an auth provider step


class TestExecutionResult(BaseModel):
    """Result of the entire test execution."""

    total_steps: int
    passed: int
    failed: int
    skipped: int
    results: list[StepResult] = Field(default_factory=list)


@dataclass
class ExecutionContext:
    """Context maintained during test execution."""

    http_client: httpx.AsyncClient
    base_url: str
    step_responses: dict[str, Any] = field(default_factory=dict)
    auth_tokens: dict[str, str] = field(default_factory=dict)  # step_id -> token


def resolve_placeholders(value: Any, step_responses: dict[str, Any]) -> Any:
    """
    Resolve placeholders like {{step_id.field_path}} with actual values from previous responses.
    """
    if isinstance(value, str):
        pattern = r"\{\{(\w+)\.([^}]+)\}\}"
        matches = re.findall(pattern, value)
        for step_id, field_path in matches:
            if step_id in step_responses:
                response = step_responses[step_id]
                # Navigate the field path
                parts = field_path.split(".")
                current = response
                try:
                    for part in parts:
                        if isinstance(current, dict):
                            current = current[part]
                        elif isinstance(current, list) and part.isdigit():
                            current = current[int(part)]
                        else:
                            current = getattr(current, part)
                    placeholder = f"{{{{{step_id}.{field_path}}}}}"
                    value = value.replace(placeholder, str(current))
                except (KeyError, IndexError, AttributeError):
                    pass  # Keep placeholder if resolution fails
        return value
    elif isinstance(value, dict):
        return {k: resolve_placeholders(v, step_responses) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_placeholders(v, step_responses) for v in value]
    return value


def build_url(
    base_url: str,
    endpoint: str,
    path_params: dict[str, Any],
    query_params: dict[str, Any],
) -> str:
    """Build the full URL with path and query parameters."""
    # Substitute path parameters
    url = endpoint
    for key, value in path_params.items():
        if value is not None:
            url = url.replace(f"{{{key}}}", str(value))

    # Build full URL
    full_url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"

    # Add query parameters (filter out None values and convert to strings)
    if query_params:
        filtered_params = {k: str(v) for k, v in query_params.items() if v is not None}
        if filtered_params:
            query_string = "&".join(f"{k}={v}" for k, v in filtered_params.items())
            full_url = f"{full_url}?{query_string}"

    return full_url


def prepare_request_body(
    body_format: BodyFormat,
    request_body: dict | None,
) -> tuple[dict | None, dict | None, dict[str, str]]:
    """
    Prepare the request body based on the body format.

    Returns:
        Tuple of (json_body, form_data, extra_headers)
    """
    if request_body is None or body_format == BodyFormat.NONE:
        return None, None, {}

    if body_format == BodyFormat.JSON:
        return request_body, None, {"Content-Type": "application/json"}

    elif body_format == BodyFormat.FORM_URLENCODED:
        # Convert all values to strings for form data, filter out None values
        form_data = {k: str(v) for k, v in request_body.items() if v is not None}
        return None, form_data, {"Content-Type": "application/x-www-form-urlencoded"}

    elif body_format == BodyFormat.MULTIPART:
        # For multipart, we need special handling
        # httpx handles this with the `files` parameter, but for now we'll use form data
        form_data = {k: str(v) for k, v in request_body.items() if v is not None}
        return None, form_data, {}  # Let httpx set the Content-Type for multipart

    elif body_format == BodyFormat.RAW:
        # Raw body - just pass as-is, treated as JSON for now
        return request_body, None, {}

    return None, None, {}


def extract_token_from_response(
    response_body: Any, token_path: str | None
) -> str | None:
    """
    Extract authentication token from response using the specified path.

    Args:
        response_body: The response body (dict)
        token_path: Dot-separated path to the token (e.g., "access_token" or "data.token")

    Returns:
        The extracted token or None if not found
    """
    if token_path is None or not isinstance(response_body, dict):
        # Try common token paths as fallback
        if isinstance(response_body, dict):
            for key in ["access_token", "token", "accessToken"]:
                if key in response_body:
                    return str(response_body[key])
        return None

    parts = token_path.split(".")
    current = response_body
    try:
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            else:
                return None
        return str(current) if current else None
    except (KeyError, TypeError):
        return None


async def execute_step(
    step: TestStep,
    payload: MockedPayload | None,
    ctx: ExecutionContext,
) -> StepResult:
    """Execute a single test step."""
    import time

    start_time = time.time()

    try:
        # Resolve any placeholders in the payload
        path_params = {}
        query_params = {}
        request_body = None
        headers = {}

        if payload:
            path_params = resolve_placeholders(payload.path_params, ctx.step_responses)
            query_params = resolve_placeholders(
                payload.query_params, ctx.step_responses
            )
            request_body = resolve_placeholders(
                payload.request_body, ctx.step_responses
            )
            headers = resolve_placeholders(payload.headers, ctx.step_responses)

        url = build_url(ctx.base_url, step.endpoint, path_params, query_params)

        # Prepare body based on format
        json_body, form_data, content_headers = prepare_request_body(
            step.body_format, request_body
        )

        # Merge headers (payload headers take precedence over content headers)
        final_headers = {**content_headers, **(headers if headers else {})}

        # Make the request with appropriate body format
        if form_data is not None:
            # Form-urlencoded or multipart
            response = await ctx.http_client.request(
                method=step.method,
                url=url,
                data=form_data,
                headers=final_headers if final_headers else None,
            )
        else:
            # JSON or no body
            response = await ctx.http_client.request(
                method=step.method,
                url=url,
                json=json_body if json_body else None,
                headers=final_headers if final_headers else None,
            )

        duration_ms = (time.time() - start_time) * 1000

        # Try to parse response as JSON
        try:
            response_body = response.json()
        except Exception:
            response_body = response.text

        # Store response for dependent steps
        ctx.step_responses[step.id] = response_body

        # Extract token if this is an auth provider step
        auth_token = None
        if (
            step.auth_requirement == AuthRequirement.AUTH_PROVIDER
            and response.status_code == step.expected_status
        ):
            auth_token = extract_token_from_response(
                response_body, step.auth_token_path
            )
            if auth_token:
                ctx.auth_tokens[step.id] = auth_token

        success = response.status_code == step.expected_status

        return StepResult(
            step_id=step.id,
            success=success,
            status_code=response.status_code,
            expected_status=step.expected_status,
            response_body=response_body,
            duration_ms=duration_ms,
            auth_token_extracted=auth_token,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return StepResult(
            step_id=step.id,
            success=False,
            expected_status=step.expected_status,
            error=str(e),
            duration_ms=duration_ms,
        )


def get_execution_order(steps: list[TestStep]) -> list[list[TestStep]]:
    """
    Determine the execution order based on dependencies.
    Returns a list of batches, where each batch can be executed in parallel.
    """
    step_map = {step.id: step for step in steps}
    completed = set()
    batches = []
    remaining = set(step.id for step in steps)

    while remaining:
        # Find steps whose dependencies are all completed
        batch = []
        for step_id in list(remaining):
            step = step_map[step_id]
            if all(dep in completed for dep in step.depends_on):
                batch.append(step)

        if not batch:
            # Circular dependency or missing dependency - add remaining as final batch
            batch = [step_map[step_id] for step_id in remaining]
            batches.append(batch)
            break

        for step in batch:
            remaining.remove(step.id)
            completed.add(step.id)

        batches.append(batch)

    return batches


async def run_test_execution_phase(
    test_plan: TestPlan,
    test_data: GeneratedTestData,
    http_client: httpx.AsyncClient | None = None,
) -> TestExecutionResult:
    """
    Run the test execution phase: execute all test steps with the generated data.

    Args:
        test_plan: The test plan to execute
        test_data: The generated mock data for each step
        http_client: Optional HTTP client (will create one if not provided)

    Returns:
        TestExecutionResult with results for all steps
    """

    # Create payload lookup
    payload_map = {p.step_id: p for p in test_data.payloads}

    async def _run_with_client(client: httpx.AsyncClient) -> TestExecutionResult:
        ctx = ExecutionContext(http_client=client, base_url=test_plan.base_url)

        results: list[StepResult] = []
        passed = 0
        failed = 0
        skipped = 0

        # Get execution order (batches of steps that can run in parallel)
        batches = get_execution_order(test_plan.steps)

        for batch in batches:
            # Execute batch (could be parallelized, but running sequentially for now for simpler debugging)
            for step in batch:
                # Check if dependencies passed
                deps_passed = all(
                    any(r.step_id == dep and r.success for r in results)
                    for dep in step.depends_on
                )

                if not deps_passed and step.depends_on:
                    # Skip this step if dependencies failed
                    results.append(
                        StepResult(
                            step_id=step.id,
                            success=False,
                            expected_status=step.expected_status,
                            error="Skipped due to failed dependencies",
                        )
                    )
                    skipped += 1
                    continue

                payload = payload_map.get(step.id)
                result = await execute_step(step, payload, ctx)
                results.append(result)

                if result.success:
                    passed += 1
                else:
                    failed += 1

        return TestExecutionResult(
            total_steps=len(test_plan.steps),
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
        )

    if http_client:
        return await _run_with_client(http_client)
    else:
        async with httpx.AsyncClient() as client:
            return await _run_with_client(client)
