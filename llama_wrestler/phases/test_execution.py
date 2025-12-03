from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
from typing import Any
from urllib.parse import quote
import httpx
import re
import time
from pydantic import BaseModel, Field
from llama_wrestler.models import APIPlan, APIStep, BodyFormat, AuthRequirement

from llama_wrestler.phases.data_generation import GeneratedTestData, MockedPayload
from llama_wrestler.schema import (
    OpenAPISchemaParser,
    RequestResponseValidator,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class StepResult(BaseModel):
    """Result of executing a single test step."""

    step_id: str
    success: bool
    status_code: int | None = None
    expected_status: int
    response_body: Any = None
    response_headers: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0
    auth_token_extracted: str | None = None  # Token if this was an auth provider step
    request_validation: ValidationResult | None = None  # Request body validation result
    response_validation: ValidationResult | None = (
        None  # Response body validation result
    )


class APIExecutionResult(BaseModel):
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
    validator: RequestResponseValidator | None = (
        None  # Optional validator for OpenAPI validation
    )


@dataclass
class PreparedRequestBody:
    """Preprocessed request body pieces ready for httpx."""

    json: Any | None = None
    data: dict[str, str] | None = None
    files: dict[str, Any] | None = None
    content: bytes | str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def choose_auth_token(step: APIStep, ctx: ExecutionContext) -> str | None:
    """Select an auth token for a step, preferring dependency providers."""
    for dep in step.depends_on:
        if dep in ctx.auth_tokens:
            return ctx.auth_tokens[dep]

    if ctx.auth_tokens:
        return next(reversed(ctx.auth_tokens.values()))

    return None


def resolve_placeholders(value: Any, step_responses: dict[str, Any]) -> Any:
    """
    Resolve placeholders like {{step_id.field_path}} with actual values from previous responses.
    """
    placeholder_pattern = re.compile(r"\{\{([\w-]+)\.([^}]+)\}\}")

    def _resolve_from_response(step_id: str, field_path: str) -> Any:
        response = step_responses[step_id]
        current: Any = response
        for part in field_path.split("."):
            if isinstance(current, dict):
                if part not in current:
                    raise KeyError(part)
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                raise KeyError(part)

        return current

    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            step_id, field_path = match.group(1), match.group(2)
            if step_id not in step_responses:
                return match.group(0)

            try:
                resolved = _resolve_from_response(step_id, field_path)
            except (KeyError, IndexError, TypeError):
                return match.group(0)

            return str(resolved)

        return placeholder_pattern.sub(_replace, value)
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
            url = url.replace(f"{{{key}}}", quote(str(value), safe=""))

    # Build full URL
    full_url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"

    # Add query parameters (filter out None values and convert to strings)
    if query_params:
        filtered_params = {k: v for k, v in query_params.items() if v is not None}
        if filtered_params:
            full_url = str(httpx.URL(full_url).copy_merge_params(filtered_params))

    return full_url


def prepare_request_body(
    body_format: BodyFormat,
    request_body: Any | None,
) -> PreparedRequestBody:
    """
    Prepare the request body based on the body format.

    Returns:
        PreparedRequestBody containing the correct pieces for httpx
    """
    if request_body is None or body_format == BodyFormat.NONE:
        return PreparedRequestBody()

    if body_format == BodyFormat.JSON:
        return PreparedRequestBody(
            json=request_body, headers={"Content-Type": "application/json"}
        )

    elif body_format == BodyFormat.FORM_URLENCODED:
        # Convert all values to strings for form data, filter out None values
        form_data = {k: str(v) for k, v in request_body.items() if v is not None}
        return PreparedRequestBody(
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    elif body_format == BodyFormat.MULTIPART:
        # Use files to let httpx set Content-Type properly
        files = {k: (None, str(v)) for k, v in request_body.items() if v is not None}
        return PreparedRequestBody(files=files)

    elif body_format == BodyFormat.RAW:
        content: bytes | str
        if isinstance(request_body, (bytes, bytearray)):
            content = bytes(request_body)
        else:
            content = str(request_body)
        return PreparedRequestBody(content=content)

    return PreparedRequestBody()


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
    step: APIStep,
    payload: MockedPayload | None,
    ctx: ExecutionContext,
) -> StepResult:
    """Execute a single test step."""
    start_time = time.perf_counter()

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
        prepared_body = prepare_request_body(step.body_format, request_body)

        # Merge headers (payload headers take precedence over content headers)
        final_headers = {
            **(prepared_body.headers if prepared_body.headers else {}),
            **(headers if headers else {}),
        }

        if step.auth_requirement in {
            AuthRequirement.REQUIRED,
            AuthRequirement.OPTIONAL,
        }:
            token = choose_auth_token(step, ctx)
            if token and "Authorization" not in final_headers:
                final_headers["Authorization"] = f"Bearer {token}"

        request_kwargs: dict[str, Any] = {
            "method": step.method,
            "url": url,
            "headers": final_headers or None,
        }
        if prepared_body.json is not None:
            request_kwargs["json"] = prepared_body.json
        if prepared_body.data is not None:
            request_kwargs["data"] = prepared_body.data
        if prepared_body.files is not None:
            request_kwargs["files"] = prepared_body.files
        if prepared_body.content is not None:
            request_kwargs["content"] = prepared_body.content

        response = await ctx.http_client.request(**request_kwargs)

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Try to parse response as JSON
        try:
            response_body = response.json()
        except Exception:
            response_body = response.text

        # Store response for dependent steps
        ctx.step_responses[step.id] = response_body

        # Validate request body if validator is available
        request_validation = None
        if ctx.validator and request_body is not None:
            request_validation = ctx.validator.validate_request_body(
                step.endpoint, step.method.lower(), request_body
            )

        # Validate response body if validator is available
        response_validation = None
        if ctx.validator and response_body is not None:
            response_validation = ctx.validator.validate_response(
                step.endpoint, step.method.lower(), response.status_code, response_body
            )

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
            response_headers=dict(response.headers),
            duration_ms=duration_ms,
            auth_token_extracted=auth_token,
            request_validation=request_validation,
            response_validation=response_validation,
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return StepResult(
            step_id=step.id,
            success=False,
            expected_status=step.expected_status,
            error=f"{e.__class__.__name__}: {e}",
            duration_ms=duration_ms,
        )


def sanitize_dependencies(steps: list[APIStep]) -> tuple[list[APIStep], set[str]]:
    """
    Remove invalid dependency references from steps.

    LLM-generated test plans sometimes reference step IDs that don't exist.
    This function filters out those invalid references and returns sanitized steps.

    Args:
        steps: List of API steps with potentially invalid dependencies

    Returns:
        Tuple of (sanitized steps, set of removed dependency IDs)
    """
    step_ids = {step.id for step in steps}
    removed_deps: set[str] = set()
    sanitized_steps: list[APIStep] = []

    for step in steps:
        invalid_deps = {dep for dep in step.depends_on if dep not in step_ids}
        if invalid_deps:
            removed_deps.update(invalid_deps)
            # Create a new step with only valid dependencies
            valid_deps = [dep for dep in step.depends_on if dep in step_ids]
            sanitized_step = step.model_copy(update={"depends_on": valid_deps})
            sanitized_steps.append(sanitized_step)
        else:
            sanitized_steps.append(step)

    if removed_deps:
        logger.warning(
            "Removed invalid dependency references from test plan: %s",
            ", ".join(sorted(removed_deps)),
        )

    return sanitized_steps, removed_deps


def get_execution_order(steps: list[APIStep]) -> list[list[APIStep]]:
    """
    Determine the execution order based on dependencies.
    Returns a list of batches, where each batch can be executed in parallel.

    Invalid dependencies are automatically removed with a warning.
    """
    # Sanitize dependencies first - remove any that reference non-existent steps
    sanitized_steps, _ = sanitize_dependencies(steps)

    step_map = {step.id: step for step in sanitized_steps}
    step_ids = set(step_map.keys())

    indegree: dict[str, int] = {step_id: 0 for step_id in step_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for step in sanitized_steps:
        for dep in step.depends_on:
            adjacency[dep].append(step.id)
            indegree[step.id] += 1

    queue: deque[str] = deque([sid for sid, deg in indegree.items() if deg == 0])
    batches: list[list[APIStep]] = []
    processed = 0

    while queue:
        batch_size = len(queue)
        batch_ids = [queue.popleft() for _ in range(batch_size)]
        batches.append([step_map[sid] for sid in batch_ids])
        processed += batch_size

        for sid in batch_ids:
            for neighbor in adjacency.get(sid, []):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

    if processed != len(sanitized_steps):
        raise ValueError("Circular dependencies detected in test plan")

    return batches


async def run_test_execution_phase(
    test_plan: APIPlan,
    test_data: GeneratedTestData,
    http_client: httpx.AsyncClient | None = None,
    openapi_spec: dict[str, Any] | None = None,
) -> APIExecutionResult:
    """
    Run the test execution phase: execute all test steps with the generated data.

    Args:
        test_plan: The test plan to execute
        test_data: The generated mock data for each step
        http_client: Optional HTTP client (will create one if not provided)
        openapi_spec: Optional OpenAPI specification for request/response validation

    Returns:
        APIExecutionResult with results for all steps
    """

    # Create payload lookup
    payload_map = {p.step_id: p for p in test_data.payloads}

    # Create validator if OpenAPI spec is provided
    validator: RequestResponseValidator | None = None
    if openapi_spec:
        parser = OpenAPISchemaParser(openapi_spec)
        validator = RequestResponseValidator(parser)

    async def _run_with_client(client: httpx.AsyncClient) -> APIExecutionResult:
        ctx = ExecutionContext(
            http_client=client,
            base_url=test_plan.base_url,
            validator=validator,
        )

        results: list[StepResult] = []
        passed = 0
        failed = 0
        skipped = 0

        # Get execution order (batches of steps that can run in parallel)
        try:
            batches = get_execution_order(test_plan.steps)
        except ValueError as err:
            error_message = str(err)
            return APIExecutionResult(
                total_steps=len(test_plan.steps),
                passed=0,
                failed=len(test_plan.steps),
                skipped=0,
                results=[
                    StepResult(
                        step_id=step.id,
                        success=False,
                        expected_status=step.expected_status,
                        error=error_message,
                    )
                    for step in test_plan.steps
                ],
            )

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

        return APIExecutionResult(
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
