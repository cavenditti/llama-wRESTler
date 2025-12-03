"""
Refinement phase: Analyze test failures and generate refined payloads.

This module uses an LLM to analyze failed test executions and generate
corrected payloads for re-execution. It also tracks regressions across
iterations and can refine the test plan itself.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from llama_wrestler.models import APIPlan, APIStep, AuthRequirement, BodyFormat
from llama_wrestler.phases.data_generation import (
    GeneratedTestData,
    MockedPayload,
)
from llama_wrestler.phases.test_execution import (
    APIExecutionResult,
    StepResult,
)
from llama_wrestler.settings import settings
from llama_wrestler.spec_utils import (
    get_security_definitions,
    extract_security_requirements,
)

logger = logging.getLogger(__name__)


@dataclass
class IterationHistory:
    """Tracks step results and payloads across iterations for regression detection."""

    # step_id -> list of (success, payload) for each iteration
    step_history: dict[str, list[tuple[bool, MockedPayload | None]]] = field(
        default_factory=dict
    )
    # step_id -> last known working payload
    last_working_payloads: dict[str, MockedPayload] = field(default_factory=dict)

    def record_iteration(
        self,
        execution_result: APIExecutionResult,
        test_data: GeneratedTestData,
    ) -> None:
        """Record results from an iteration."""
        payload_map = {p.step_id: p for p in test_data.payloads}
        result_map = {r.step_id: r for r in execution_result.results}

        for step_id, result in result_map.items():
            payload = payload_map.get(step_id)

            if step_id not in self.step_history:
                self.step_history[step_id] = []

            self.step_history[step_id].append((result.success, payload))

            # Track last working payload
            if result.success and payload:
                self.last_working_payloads[step_id] = payload

    def get_regressions(self) -> list[str]:
        """Get step IDs that regressed (were passing but now failing)."""
        regressions = []
        for step_id, history in self.step_history.items():
            if len(history) >= 2:
                # Check if it was passing before but failing now
                prev_success = history[-2][0]
                curr_success = history[-1][0]
                if prev_success and not curr_success:
                    regressions.append(step_id)
        return regressions

    def get_revert_payloads(self, regressions: list[str]) -> dict[str, MockedPayload]:
        """Get the last working payloads for regressed steps."""
        return {
            step_id: self.last_working_payloads[step_id]
            for step_id in regressions
            if step_id in self.last_working_payloads
        }


class RefinedPayload(BaseModel):
    """A refined payload for a failed step."""

    step_id: str = Field(description="The step ID being refined")
    request_body: dict | list | None = Field(None, description="Corrected request body")
    path_params: dict[str, Any] = Field(
        default_factory=dict, description="Corrected path parameters"
    )
    query_params: dict[str, Any] = Field(
        default_factory=dict, description="Corrected query parameters"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Corrected headers"
    )
    reasoning: str = Field(
        description="Explanation of what was wrong and how it was fixed"
    )


class RefinedStep(BaseModel):
    """A refined test step definition."""

    step_id: str = Field(description="The step ID being refined")
    auth_requirement: AuthRequirement | None = Field(
        None, description="Corrected auth requirement"
    )
    expected_status: int | None = Field(
        None, description="Corrected expected status code"
    )
    body_format: BodyFormat | None = Field(None, description="Corrected body format")
    depends_on: list[str] | None = Field(None, description="Corrected dependencies")
    reasoning: str = Field(
        description="Explanation of what was wrong and how it was fixed"
    )


class RefinementResult(BaseModel):
    """Result of the refinement phase."""

    refined_payloads: list[RefinedPayload] = Field(
        default_factory=list, description="List of refined payloads"
    )
    refined_steps: list[RefinedStep] = Field(
        default_factory=list, description="List of refined test step definitions"
    )
    analysis_summary: str = Field(
        description="Summary of failure analysis and fixes applied"
    )


def fix_auth_requirements_from_spec(test_plan: APIPlan, openapi_spec: dict) -> APIPlan:
    """
    Fix auth requirements in the test plan based on the OpenAPI spec.

    This addresses warnings like "has auth_requirement=NONE but spec requires: OAuth2PasswordBearer"
    by updating the test plan steps to match the spec's security requirements.

    Args:
        test_plan: The test plan to fix
        openapi_spec: The OpenAPI specification

    Returns:
        Updated APIPlan with corrected auth requirements
    """
    spec_requirements = extract_security_requirements(openapi_spec)
    security_defs = get_security_definitions(openapi_spec)

    updated_steps = []
    for step in test_plan.steps:
        # Build the endpoint key to look up in spec
        endpoint_key = f"{step.method.upper()} {step.endpoint}"

        # Also try to match path parameter patterns
        matching_key = None
        for spec_key in spec_requirements:
            spec_method, spec_path = spec_key.split(" ", 1)
            if spec_method != step.method.upper():
                continue

            if spec_path == step.endpoint:
                matching_key = spec_key
                break

            # Convert path params like {id} to regex pattern
            pattern = re.sub(r"\{[^}]+\}", r"[^/]+", spec_path)
            pattern = f"^{pattern}$"
            if re.match(pattern, step.endpoint):
                matching_key = spec_key
                break

        if matching_key is None:
            updated_steps.append(step)
            continue

        required_schemes = spec_requirements[matching_key]

        # Check if we need to update auth requirement
        needs_update = False
        new_auth_requirement = step.auth_requirement

        if required_schemes and step.auth_requirement == AuthRequirement.NONE:
            # Spec requires auth but step says NONE - fix it
            new_auth_requirement = AuthRequirement.REQUIRED
            needs_update = True
            logger.info(
                "Fixing auth_requirement for step '%s': NONE -> REQUIRED (spec requires: %s)",
                step.id,
                ", ".join(required_schemes),
            )

        if needs_update:
            # Create updated step
            updated_step = APIStep(
                id=step.id,
                description=step.description,
                endpoint=step.endpoint,
                method=step.method,
                depends_on=step.depends_on,
                payload_description=step.payload_description,
                expected_status=step.expected_status,
                body_format=step.body_format,
                auth_requirement=new_auth_requirement,
                auth_token_path=step.auth_token_path,
            )
            updated_steps.append(updated_step)
        else:
            updated_steps.append(step)

    return APIPlan(
        summary=test_plan.summary,
        base_url=test_plan.base_url,
        steps=updated_steps,
    )


@dataclass
class RefinementDeps:
    """Dependencies for the refinement agent."""

    test_plan: APIPlan
    openapi_spec: dict
    execution_result: APIExecutionResult
    current_test_data: GeneratedTestData
    failed_steps: list[APIStep]


REFINEMENT_SYSTEM_PROMPT = """
You are an expert API testing analyst. Your job is to analyze failed API test executions
and generate corrected payloads that will succeed.

You will receive:
1. Failed test step definitions
2. The original payloads that were used
3. The error responses and status codes received
4. The OpenAPI specification for reference

Your task is to:
1. Analyze WHY each test failed (wrong data format, missing required fields, invalid values, etc.)
2. Generate CORRECTED payloads that should succeed
3. Explain your reasoning for each fix

COMMON FAILURE PATTERNS:

1. **400 Bad Request**: Usually means:
   - Missing required fields in request body
   - Wrong data types (string vs int, etc.)
   - Invalid enum values
   - Constraint violations (min/max, pattern, etc.)

2. **401 Unauthorized**: Usually means:
   - Missing or malformed Authorization header
   - Token placeholder not being resolved
   - Wrong auth scheme (Bearer vs Basic, etc.)

3. **404 Not Found**: Usually means:
   - Path parameter has wrong value
   - Resource from dependency doesn't exist (cascade failure)
   - Wrong endpoint path

4. **422 Unprocessable Entity**: Usually means:
   - Validation errors in the request body
   - Business logic constraints violated
   - Related resources don't exist (foreign key issues)

5. **500 Internal Server Error**: Usually means:
   - Server bug OR
   - Invalid data causing server-side crash
   - Check if request body schema is correct

PLACEHOLDER SYNTAX:
Use {{step_id.field_path}} to reference values from previous step responses.
Examples:
- {{auth_login.access_token}} - token from login response
- {{create_user.id}} - ID from created user
- {{create_item.data.id}} - nested ID

IMPORTANT RULES:
1. Keep working placeholders - don't replace them with literal values
2. Fix data types to match OpenAPI schema
3. Add missing required fields
4. Use valid enum values from the schema
5. For cascade failures (dependency failed), note that the step may need re-running after fix

OUTPUT FORMAT:

For PAYLOAD issues (refined_payloads):
- step_id: The step identifier
- request_body: The corrected body (or null if no body)
- path_params: Corrected path parameters
- query_params: Corrected query parameters
- headers: Corrected headers (especially Authorization)
- reasoning: Clear explanation of what was wrong and how you fixed it

For TEST PLAN issues (refined_steps):
If the test step definition itself is wrong (wrong expected_status, missing auth_requirement, wrong body_format),
include entries in refined_steps:
- step_id: The step identifier
- auth_requirement: Corrected auth requirement if wrong (none, optional, required, auth_provider)
- expected_status: Corrected expected status code if wrong
- body_format: Corrected body format if wrong (json, form_urlencoded, multipart, none, raw)
- depends_on: Corrected dependencies if wrong
- reasoning: Clear explanation of what was wrong and how you fixed it

6. **BROKEN DEPENDENCIES**: When steps reference non-existent step IDs in depends_on:
   - These MUST be fixed in refined_steps
   - Either correct the step ID to reference an existing step
   - Or set depends_on to an empty list [] if no valid dependency exists
   - Look at the available step IDs in the test plan to find valid alternatives
   - This is CRITICAL: steps with invalid dependencies will never execute correctly

IMPORTANT: Only include refined_steps when the TEST PLAN is wrong, not the payload.
For example, if a test expects status 200 but the correct response is 201, add a refined_step.
If auth is required but the step has auth_requirement=none, add a refined_step.
If a step has invalid dependencies that don't exist, add a refined_step with corrected depends_on.
"""


def _create_refinement_agent() -> Agent[RefinementDeps, RefinementResult]:
    """Create the refinement agent."""
    return Agent(
        model=f"openai:{settings.openai_model}",
        deps_type=RefinementDeps,
        output_type=RefinementResult,
        system_prompt=REFINEMENT_SYSTEM_PROMPT,
    )


def _get_failed_steps(
    test_plan: APIPlan, execution_result: APIExecutionResult
) -> list[tuple[APIStep, StepResult, MockedPayload | None]]:
    """Get failed steps with their results and original payloads."""
    step_map = {s.id: s for s in test_plan.steps}

    failed = []
    for result in execution_result.results:
        if not result.success and result.error != "Skipped due to failed dependencies":
            step = step_map.get(result.step_id)
            if step:
                failed.append((step, result, None))

    return failed


def _get_steps_with_broken_deps(
    test_plan: APIPlan, execution_result: APIExecutionResult
) -> list[tuple[APIStep, list[str]]]:
    """Get steps that had invalid dependencies removed.

    Returns:
        List of (step, list of removed dependency IDs)
    """
    step_map = {s.id: s for s in test_plan.steps}
    steps_with_issues = []

    for step_id, removed_deps in execution_result.steps_with_broken_deps.items():
        step = step_map.get(step_id)
        if step:
            steps_with_issues.append((step, removed_deps))

    return steps_with_issues


def _build_failure_context(
    failed_steps: list[tuple[APIStep, StepResult, MockedPayload | None]],
    test_data: GeneratedTestData,
    openapi_spec: dict,
    steps_with_broken_deps: list[tuple[APIStep, list[str]]] | None = None,
) -> str:
    """Build a detailed context string for the LLM about failures.

    Args:
        failed_steps: List of (step, result, payload) tuples for failed steps
        test_data: The test data used
        openapi_spec: The OpenAPI specification
        steps_with_broken_deps: Optional list of (step, removed_dep_ids) for steps with invalid dependencies
    """
    payload_map = {p.step_id: p for p in test_data.payloads}
    paths = openapi_spec.get("paths", {})

    # Get schema definitions
    if "swagger" in openapi_spec:
        definitions = openapi_spec.get("definitions", {})
    else:
        definitions = openapi_spec.get("components", {}).get("schemas", {})

    parts = ["# Failed Test Steps Analysis\n"]

    # Add section for steps with broken dependencies
    if steps_with_broken_deps:
        parts.append("\n## ⚠️ Steps with Invalid Dependencies\n")
        parts.append(
            "The following steps reference dependencies that don't exist in the test plan."
        )
        parts.append("These dependencies need to be fixed by either:")
        parts.append("1. Removing the dependency if it's not needed")
        parts.append("2. Correcting the dependency ID to reference an existing step")
        parts.append(
            "3. The step may need its depends_on field cleared if no valid dependency exists\n"
        )

        for step, removed_deps in steps_with_broken_deps:
            parts.append(f"\n### Step: {step.id}")
            parts.append(f"- Description: {step.description}")
            parts.append(f"- Endpoint: {step.method} {step.endpoint}")
            parts.append(f"- Current depends_on: {step.depends_on}")
            parts.append(f"- **Invalid dependencies removed**: {removed_deps}")
            parts.append(
                "- **ACTION REQUIRED**: Include this step in refined_steps with corrected depends_on"
            )

        parts.append("\n---\n")

    for step, result, _ in failed_steps:
        payload = payload_map.get(step.id)

        parts.append(f"\n## Step: {step.id}")
        parts.append(f"- Description: {step.description}")
        parts.append(f"- Endpoint: {step.method} {step.endpoint}")
        parts.append(f"- Expected Status: {step.expected_status}")
        parts.append(f"- Actual Status: {result.status_code}")
        parts.append(f"- Error: {result.error or 'None'}")

        if result.response_body:
            try:
                resp_str = json.dumps(result.response_body, indent=2)
                if len(resp_str) > 1000:
                    resp_str = resp_str[:1000] + "\n... (truncated)"
                parts.append(f"- Response Body:\n```json\n{resp_str}\n```")
            except Exception:
                parts.append(f"- Response Body: {result.response_body}")

        if payload:
            parts.append("\n### Original Payload:")
            parts.append(f"- Request Body: {json.dumps(payload.request_body)}")
            parts.append(f"- Path Params: {json.dumps(payload.path_params)}")
            parts.append(f"- Query Params: {json.dumps(payload.query_params)}")
            parts.append(f"- Headers: {json.dumps(payload.headers)}")

        # Add relevant OpenAPI spec
        path_spec = paths.get(step.endpoint, {})
        method_spec = path_spec.get(step.method.lower(), {})
        if method_spec:
            parts.append("\n### OpenAPI Spec for this endpoint:")
            spec_str = json.dumps(method_spec, indent=2)
            if len(spec_str) > 2000:
                spec_str = spec_str[:2000] + "\n... (truncated)"
            parts.append(f"```json\n{spec_str}\n```")

        parts.append("\n---")

    # Add schema definitions (limited)
    if definitions:
        parts.append("\n## Relevant Schema Definitions:")
        def_str = json.dumps(definitions, indent=2)
        if len(def_str) > 3000:
            def_str = def_str[:3000] + "\n... (truncated)"
        parts.append(f"```json\n{def_str}\n```")

    return "\n".join(parts)


async def run_refinement_phase(
    test_plan: APIPlan,
    openapi_spec: dict,
    execution_result: APIExecutionResult,
    test_data: GeneratedTestData,
    history: IterationHistory | None = None,
) -> tuple[APIPlan, GeneratedTestData, RefinementResult]:
    """
    Run the refinement phase to fix failed test payloads and test plan.

    Args:
        test_plan: The test plan
        openapi_spec: The OpenAPI specification
        execution_result: Results from test execution
        test_data: The current test data that was used
        history: Optional iteration history for regression detection

    Returns:
        Tuple of (updated APIPlan, updated GeneratedTestData, RefinementResult with analysis)
    """
    # First, record this iteration in history if provided
    if history:
        history.record_iteration(execution_result, test_data)

    # Identify truly failed steps (not skipped due to dependencies)
    failed_step_results = _get_failed_steps(test_plan, execution_result)

    # Identify steps with broken dependencies that need fixing
    steps_with_broken_deps = _get_steps_with_broken_deps(test_plan, execution_result)

    if steps_with_broken_deps:
        logger.warning(
            "Found %d steps with invalid dependencies that need refinement: %s",
            len(steps_with_broken_deps),
            ", ".join(s.id for s, _ in steps_with_broken_deps),
        )

    if not failed_step_results and not steps_with_broken_deps:
        logger.info("No fixable failures found - all failures are due to dependencies")
        return (
            test_plan,
            test_data,
            RefinementResult(
                refined_payloads=[],
                refined_steps=[],
                analysis_summary="No fixable failures. All failed steps were due to dependency failures.",
            ),
        )

    failed_step_ids = [s.id for s, _, _ in failed_step_results]
    failed_steps = [s for s, _, _ in failed_step_results]
    broken_dep_step_ids = [s.id for s, _ in steps_with_broken_deps]

    logger.info(
        "Analyzing %d failed steps and %d steps with broken dependencies for refinement",
        len(failed_step_ids),
        len(broken_dep_step_ids),
    )

    # Build context for the LLM
    failure_context = _build_failure_context(
        failed_step_results, test_data, openapi_spec, steps_with_broken_deps
    )

    # Create and run the agent
    agent = _create_refinement_agent()
    deps = RefinementDeps(
        test_plan=test_plan,
        openapi_spec=openapi_spec,
        execution_result=execution_result,
        current_test_data=test_data,
        failed_steps=failed_steps,
    )

    prompt = f"""Analyze the following failed test steps and generate corrected payloads.

{failure_context}

Generate refined payloads for these failed steps: {failed_step_ids}

{"IMPORTANT: The following steps have INVALID DEPENDENCIES that reference non-existent steps: " + str(broken_dep_step_ids) + ". You MUST include these in refined_steps with corrected depends_on values (either fix the dependency IDs or set to empty list if no valid dependency exists)." if broken_dep_step_ids else ""}

Focus on fixing the root causes of the failures based on the error responses and OpenAPI schema.
If the test plan step definition itself is wrong (wrong expected_status, missing auth_requirement, wrong depends_on, etc.),
also include entries in refined_steps.
"""

    result = await agent.run(prompt, deps=deps)
    refinement_result = result.output

    logger.info(
        "Refinement complete: %d payloads refined, %d steps refined",
        len(refinement_result.refined_payloads),
        len(refinement_result.refined_steps),
    )

    # Apply test plan refinements
    updated_test_plan = _apply_step_refinements(
        test_plan, refinement_result.refined_steps
    )

    # Merge refined payloads into test data
    updated_payloads = list(test_data.payloads)
    payload_index = {p.step_id: i for i, p in enumerate(updated_payloads)}

    for refined in refinement_result.refined_payloads:
        if refined.step_id in payload_index:
            idx = payload_index[refined.step_id]
            updated_payloads[idx] = MockedPayload(
                step_id=refined.step_id,
                request_body=refined.request_body,
                path_params=refined.path_params,
                query_params=refined.query_params,
                headers=refined.headers,
            )
            logger.debug(
                "Updated payload for %s: %s", refined.step_id, refined.reasoning
            )

    # Handle regressions - revert to last working payloads
    if history:
        regressions = history.get_regressions()
        if regressions:
            logger.warning(
                "Detected %d regressions, reverting to previous working payloads",
                len(regressions),
            )
            revert_payloads = history.get_revert_payloads(regressions)
            for step_id, working_payload in revert_payloads.items():
                if step_id in payload_index:
                    idx = payload_index[step_id]
                    updated_payloads[idx] = working_payload
                    logger.info("Reverted payload for regressed step: %s", step_id)

    updated_test_data = GeneratedTestData(payloads=updated_payloads)

    return updated_test_plan, updated_test_data, refinement_result


def _apply_step_refinements(
    test_plan: APIPlan, refined_steps: list[RefinedStep]
) -> APIPlan:
    """Apply refined step definitions to the test plan."""
    if not refined_steps:
        return test_plan

    step_updates = {rs.step_id: rs for rs in refined_steps}
    updated_steps = []

    for step in test_plan.steps:
        if step.id in step_updates:
            refined = step_updates[step.id]
            logger.info(
                "Applying step refinement for %s: %s", step.id, refined.reasoning
            )

            updated_step = APIStep(
                id=step.id,
                description=step.description,
                endpoint=step.endpoint,
                method=step.method,
                depends_on=refined.depends_on
                if refined.depends_on is not None
                else step.depends_on,
                payload_description=step.payload_description,
                expected_status=refined.expected_status
                if refined.expected_status is not None
                else step.expected_status,
                body_format=refined.body_format
                if refined.body_format is not None
                else step.body_format,
                auth_requirement=refined.auth_requirement
                if refined.auth_requirement is not None
                else step.auth_requirement,
                auth_token_path=step.auth_token_path,
            )
            updated_steps.append(updated_step)
        else:
            updated_steps.append(step)

    return APIPlan(
        summary=test_plan.summary,
        base_url=test_plan.base_url,
        steps=updated_steps,
    )


def get_refinable_failure_count(execution_result: APIExecutionResult) -> int:
    """
    Count failures that are potentially refinable (not dependency failures).
    """
    count = 0
    for result in execution_result.results:
        if not result.success and result.error != "Skipped due to failed dependencies":
            count += 1
    return count


def calculate_pass_rate(execution_result: APIExecutionResult) -> float:
    """Calculate the pass rate as a percentage."""
    if execution_result.total_steps == 0:
        return 0.0
    return (execution_result.passed / execution_result.total_steps) * 100
