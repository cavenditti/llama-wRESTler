from dataclasses import dataclass, field
from typing import Any
import httpx
import re
from pydantic import BaseModel, Field

from ..models import TestPlan, TestStep
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


def build_url(base_url: str, endpoint: str, path_params: dict[str, Any], query_params: dict[str, Any]) -> str:
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
            query_params = resolve_placeholders(payload.query_params, ctx.step_responses)
            request_body = resolve_placeholders(payload.request_body, ctx.step_responses)
            headers = resolve_placeholders(payload.headers, ctx.step_responses)
        
        url = build_url(ctx.base_url, step.endpoint, path_params, query_params)
        
        # Make the request
        response = await ctx.http_client.request(
            method=step.method,
            url=url,
            json=request_body if request_body else None,
            headers=headers if headers else None,
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Try to parse response as JSON
        try:
            response_body = response.json()
        except Exception:
            response_body = response.text
        
        # Store response for dependent steps
        ctx.step_responses[step.id] = response_body
        
        success = response.status_code == step.expected_status
        
        return StepResult(
            step_id=step.id,
            success=success,
            status_code=response.status_code,
            expected_status=step.expected_status,
            response_body=response_body,
            duration_ms=duration_ms,
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
    import asyncio
    
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
                    results.append(StepResult(
                        step_id=step.id,
                        success=False,
                        expected_status=step.expected_status,
                        error="Skipped due to failed dependencies",
                    ))
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
