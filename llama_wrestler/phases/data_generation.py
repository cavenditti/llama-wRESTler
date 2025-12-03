from dataclasses import dataclass
import json
import logging
import re
from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from llama_wrestler.models import (
    APIPlan,
    APIStep,
    APICredentials,
    AuthRequirement,
)
from llama_wrestler.settings import settings
from llama_wrestler.schema import (
    OpenAPISchemaParser,
    DeterministicGenerator,
    generate_data_from_schema,
)

logger = logging.getLogger(__name__)

# Maximum number of retry passes to fill in missing payloads
MAX_DATA_GENERATION_RETRIES = 3


class MockedPayload(BaseModel):
    """Mocked data for a single test step."""

    step_id: str = Field(description="The ID of the test step this payload is for")
    request_body: dict | None = Field(
        None, description="The request body to send, if applicable"
    )
    path_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Path parameters to substitute in the endpoint URL (values will be converted to strings)",
    )
    query_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameters to append to the URL (values will be converted to strings)",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to include in the request",
    )


class GeneratedTestData(BaseModel):
    """All generated mock data for the test plan."""

    payloads: list[MockedPayload] = Field(
        description="List of mocked payloads for each test step"
    )


@dataclass
class DataGenerationDeps:
    """Dependencies for the data generation agent (bulk mode)."""

    test_plan: APIPlan
    openapi_spec: dict
    credentials: APICredentials | None = None


@dataclass
class SingleStepDeps:
    """Dependencies for per-step data generation agent."""

    step: APIStep
    openapi_spec: dict
    credentials: APICredentials | None = None
    # Context from previous steps for dependency resolution
    previous_step_ids: list[str] = None  # type: ignore
    auth_provider_step_id: str | None = None


# ============================================================================
# Per-step agent (uses weak model for efficiency)
# ============================================================================

SINGLE_STEP_SYSTEM_PROMPT = """
You are a test data generation expert. Generate mock data for a SINGLE API test step.

You will receive:
1. A single test step definition
2. Relevant parts of the OpenAPI specification
3. Context about previous steps (for dependency placeholders)

OUTPUT STRUCTURE (MockedPayload):
- step_id: The ID of the test step (copy from input)
- request_body: Dict for POST/PUT/PATCH, null for GET/DELETE
- path_params: Dict of path parameters from URL template (e.g., /items/{item_id})
- query_params: Dict of query parameters
- headers: Dict of headers (IMPORTANT for Authorization!)

BODY FORMAT RULES:
- "json": Standard JSON object
- "form_urlencoded": Dict for form data (OAuth2: include grant_type)
- "multipart": Dict with file info
- "none": request_body must be null

AUTHENTICATION RULES:
- auth_requirement="auth_provider": This IS the login endpoint. Put credentials in request_body.
- auth_requirement="required": Add Authorization header with placeholder: {"Authorization": "Bearer {{auth_step_id.access_token}}"}
- auth_requirement="none"/"optional": No auth header needed

PLACEHOLDER SYNTAX:
Use {{step_id.field_path}} to reference values from previous step responses.
Examples:
- {{auth_login.access_token}} - token from login
- {{create_item.id}} - ID from created item
- {{user_create.data.user_id}} - nested field

IMPORTANT:
1. Check endpoint for path parameters: /items/{item_id} → path_params: {"item_id": ...}
2. For auth_requirement="required", ALWAYS add Authorization header
3. Use realistic but fake data
4. If credentials are provided for auth_provider steps, USE THEM EXACTLY
"""


def _create_single_step_agent() -> Agent[SingleStepDeps, MockedPayload]:
    """Create a per-step data generation agent using the weak model."""
    return Agent(
        model=f"openai:{settings.openai_weak_model}",
        deps_type=SingleStepDeps,
        output_type=MockedPayload,
        system_prompt=SINGLE_STEP_SYSTEM_PROMPT,
    )


def _get_endpoint_spec(openapi_spec: dict, endpoint: str, method: str) -> dict:
    """Extract the relevant spec portion for a specific endpoint."""
    paths = openapi_spec.get("paths", {})
    path_spec = paths.get(endpoint, {})
    operation_spec = path_spec.get(method.lower(), {})

    # Include definitions/schemas for reference
    result = {
        "endpoint": endpoint,
        "method": method,
        "operation": operation_spec,
    }

    # Add schema definitions
    if "swagger" in openapi_spec:
        result["definitions"] = openapi_spec.get("definitions", {})
    else:
        components = openapi_spec.get("components", {})
        result["schemas"] = components.get("schemas", {})

    return result


async def _generate_single_step_data(
    step: APIStep,
    openapi_spec: dict,
    credentials: APICredentials | None,
    previous_step_ids: list[str],
    auth_provider_step_id: str | None,
) -> MockedPayload:
    """Generate data for a single step using the weak model."""
    agent = _create_single_step_agent()

    deps = SingleStepDeps(
        step=step,
        openapi_spec=openapi_spec,
        credentials=credentials,
        previous_step_ids=previous_step_ids,
        auth_provider_step_id=auth_provider_step_id,
    )

    # Build a focused prompt for this step
    endpoint_spec = _get_endpoint_spec(openapi_spec, step.endpoint, step.method)

    prompt_parts = [
        "Generate test data for this step:",
        "",
        f"Step ID: {step.id}",
        f"Description: {step.description}",
        f"Endpoint: {step.method} {step.endpoint}",
        f"Body Format: {step.body_format.value}",
        f"Auth Requirement: {step.auth_requirement.value}",
        f"Dependencies: {step.depends_on}",
        f"Expected Status: {step.expected_status}",
    ]

    if step.payload_description:
        prompt_parts.append(f"Payload Description: {step.payload_description}")

    prompt_parts.extend(
        [
            "",
            "OpenAPI Spec for this endpoint:",
            json.dumps(endpoint_spec, indent=2),
        ]
    )

    if auth_provider_step_id and step.auth_requirement == AuthRequirement.REQUIRED:
        prompt_parts.extend(
            [
                "",
                f"NOTE: Use this auth step ID for Authorization header: {auth_provider_step_id}",
            ]
        )

    if credentials and step.auth_requirement == AuthRequirement.AUTH_PROVIDER:
        prompt_parts.extend(
            [
                "",
                "CREDENTIALS TO USE (use these EXACTLY):",
                credentials.to_prompt_context(),
            ]
        )

    if previous_step_ids:
        prompt_parts.extend(
            [
                "",
                f"Previous steps (for placeholders): {previous_step_ids}",
            ]
        )

    prompt = "\n".join(prompt_parts)

    result = await agent.run(prompt, deps=deps)
    return result.output


# ============================================================================
# Bulk agent (uses strong model, legacy approach)
# ============================================================================

data_generation_agent = Agent(
    model=f"openai:{settings.openai_model}",
    deps_type=DataGenerationDeps,
    output_type=GeneratedTestData,
    system_prompt="""
You are a test data generation expert. Your goal is to generate realistic mock data for API testing.

You will be provided with:
1. A test plan containing endpoint steps with their dependencies, body formats, and auth requirements
2. An OpenAPI specification describing the API schema

Your task:
1. Analyze each test step in the plan
2. For each step, generate appropriate mock data based on the OpenAPI schema
3. Ensure data consistency across dependent steps
4. Use realistic but fake data (names, emails, etc.)
5. Follow any constraints defined in the OpenAPI spec (required fields, formats, enums, etc.)

CRITICAL - Output structure for each MockedPayload:

Each payload MUST have these fields populated correctly:
- step_id: The ID of the test step (string)
- request_body: The body for POST/PUT/PATCH requests (format depends on body_format), or null
- path_params: Dict of path parameters extracted from the endpoint URL template
- query_params: Dict of query parameters 
- headers: Dict of additional headers (IMPORTANT for auth!)

BODY FORMAT HANDLING:

Each test step has a `body_format` field that tells you how to structure the request_body:

1. body_format="json" (default):
   - Standard JSON object for the request body
   - Example: {"name": "John", "email": "john@example.com"}

2. body_format="form_urlencoded" (OAuth2, login forms):
   - Still use a dict, but it will be sent as form data
   - For OAuth2 password flow: {"username": "user@example.com", "password": "pass123", "grant_type": "password"}
   - For login: {"username": "admin", "password": "secret"}

3. body_format="multipart" (file uploads):
   - Use dict with file info, actual file handling done by executor
   - Example: {"file": "test_file.txt", "description": "Test upload"}

4. body_format="none":
   - request_body should be null

AUTHENTICATION HANDLING:

Each test step has an `auth_requirement` field:

1. auth_requirement="auth_provider":
   - This is a login/token endpoint
   - Generate credentials in request_body
   - No auth header needed (this creates the token)

2. auth_requirement="required":
   - This endpoint needs authentication
   - MUST include Authorization header with token from auth step
   - Use placeholder: {"Authorization": "Bearer {{auth_step_id.access_token}}"}
   - Check the depends_on to find the auth step ID

3. auth_requirement="none" or "optional":
   - No auth header needed

EXAMPLES:

1. OAuth2 Login (form_urlencoded, auth_provider):
{
  "step_id": "auth_login_success",
  "request_body": {"username": "admin@example.com", "password": "SecurePass123!", "grant_type": "password"},
  "path_params": {},
  "query_params": {},
  "headers": {}
}

2. Protected Endpoint (json, required auth):
{
  "step_id": "create_item",
  "request_body": {"title": "Test Item", "description": "A test item"},
  "path_params": {},
  "query_params": {},
  "headers": {"Authorization": "Bearer {{auth_login_success.access_token}}"}
}

3. GET with path param (none body, required auth):
{
  "step_id": "get_item_by_id",
  "request_body": null,
  "path_params": {"item_id": "{{create_item.id}}"},
  "query_params": {},
  "headers": {"Authorization": "Bearer {{auth_login_success.access_token}}"}
}

4. Public endpoint (no auth):
{
  "step_id": "health_check",
  "request_body": null,
  "path_params": {},
  "query_params": {},
  "headers": {}
}

5. File upload (multipart):
{
  "step_id": "upload_avatar",
  "request_body": {"file": "avatar.jpg"},
  "path_params": {},
  "query_params": {},
  "headers": {"Authorization": "Bearer {{auth_login_success.access_token}}"}
}

PLACEHOLDER SYNTAX:
Use {{step_id.field_path}} to reference values from previous step responses.
- {{auth_login_success.access_token}} - token from login response
- {{create_item.id}} - ID from created item response
- {{user_create.username}} - username from created user

IMPORTANT RULES:
1. ALWAYS check the endpoint template for path parameters (e.g., /items/{item_id})
2. ALWAYS check the OpenAPI spec for query parameters
3. ALWAYS add Authorization header for steps with auth_requirement="required"
4. For form_urlencoded OAuth2, include grant_type if the spec requires it
""",
)


@data_generation_agent.tool
async def get_test_plan(ctx: RunContext[DataGenerationDeps]) -> str:
    """Get the test plan to generate data for."""
    return ctx.deps.test_plan.model_dump_json(indent=2)


@data_generation_agent.tool
async def get_openapi_spec(ctx: RunContext[DataGenerationDeps]) -> str:
    """Get the OpenAPI specification for schema reference."""
    return json.dumps(ctx.deps.openapi_spec, indent=2)


@data_generation_agent.tool
async def get_test_credentials(ctx: RunContext[DataGenerationDeps]) -> str:
    """Get the credentials to use for authentication steps. Use these EXACT credentials for auth_provider steps."""
    if ctx.deps.credentials:
        return ctx.deps.credentials.to_prompt_context()
    return "No credentials provided - generate realistic test credentials"


# ============================================================================
# Helper functions
# ============================================================================


def _extract_path_params(endpoint: str) -> list[str]:
    """Extract path parameter names from an endpoint template."""
    return re.findall(r"\{(\w+)\}", endpoint)


def _find_auth_provider_step(
    step_id: str, test_plan: APIPlan, dependencies: list[str]
) -> str | None:
    """Find the auth provider step ID from dependencies."""
    step_map = {s.id: s for s in test_plan.steps}
    for dep_id in dependencies:
        dep = step_map.get(dep_id)
        if dep and dep.auth_requirement == AuthRequirement.AUTH_PROVIDER:
            return dep_id
    return None


# ============================================================================
# Validation and coverage functions
# ============================================================================


def get_required_step_ids(test_plan: APIPlan) -> set[str]:
    """
    Get the set of step IDs that require generated data.

    Args:
        test_plan: The test plan

    Returns:
        Set of step IDs from the test plan
    """
    return {step.id for step in test_plan.steps}


def get_generated_step_ids(test_data: GeneratedTestData) -> set[str]:
    """
    Get the set of step IDs that have generated data.

    Args:
        test_data: The generated test data

    Returns:
        Set of step IDs with payloads
    """
    return {payload.step_id for payload in test_data.payloads}


def find_missing_step_ids(
    test_plan: APIPlan, test_data: GeneratedTestData
) -> set[str]:
    """
    Find step IDs that are in the test plan but missing from generated data.

    Args:
        test_plan: The test plan
        test_data: The generated test data

    Returns:
        Set of step IDs that are missing payloads
    """
    required = get_required_step_ids(test_plan)
    generated = get_generated_step_ids(test_data)
    return required - generated


def find_extra_step_ids(
    test_plan: APIPlan, test_data: GeneratedTestData
) -> set[str]:
    """
    Find step IDs in generated data that are not in the test plan.

    Args:
        test_plan: The test plan
        test_data: The generated test data

    Returns:
        Set of step IDs that are extra (not in test plan)
    """
    required = get_required_step_ids(test_plan)
    generated = get_generated_step_ids(test_data)
    return generated - required


def filter_valid_payloads(
    test_plan: APIPlan, test_data: GeneratedTestData
) -> GeneratedTestData:
    """
    Remove payloads for steps that don't exist in the test plan.

    Args:
        test_plan: The test plan
        test_data: The generated test data

    Returns:
        GeneratedTestData with only valid payloads
    """
    valid_step_ids = get_required_step_ids(test_plan)
    valid_payloads = [
        p for p in test_data.payloads if p.step_id in valid_step_ids
    ]
    return GeneratedTestData(payloads=valid_payloads)


def merge_test_data(
    base_data: GeneratedTestData,
    additional_data: GeneratedTestData,
    test_plan: APIPlan,
) -> GeneratedTestData:
    """
    Merge additional payloads into base data, avoiding duplicates.

    Payloads from additional_data are added only if their step_id
    is not already present in base_data.

    Args:
        base_data: The base test data
        additional_data: Additional payloads to merge
        test_plan: The test plan (for ordering)

    Returns:
        Merged GeneratedTestData
    """
    existing_ids = {p.step_id for p in base_data.payloads}
    valid_step_ids = get_required_step_ids(test_plan)

    new_payloads = [
        p
        for p in additional_data.payloads
        if p.step_id not in existing_ids and p.step_id in valid_step_ids
    ]

    # Combine and sort by test plan order
    all_payloads = base_data.payloads + new_payloads
    step_order = {step.id: i for i, step in enumerate(test_plan.steps)}
    sorted_payloads = sorted(
        all_payloads,
        key=lambda p: step_order.get(p.step_id, float("inf")),
    )

    return GeneratedTestData(payloads=sorted_payloads)


def get_missing_steps(test_plan: APIPlan, test_data: GeneratedTestData) -> list[APIStep]:
    """
    Get the list of steps that are missing from the generated data.

    Args:
        test_plan: The test plan
        test_data: The generated test data

    Returns:
        List of APIStep objects that need data generation
    """
    missing_ids = find_missing_step_ids(test_plan, test_data)
    return [step for step in test_plan.steps if step.id in missing_ids]


def run_deterministic_data_generation(
    test_plan: APIPlan,
    openapi_spec: dict,
    credentials: APICredentials | None = None,
    seed: int | str | None = None,
) -> GeneratedTestData:
    """
    Generate test data deterministically using schema-based generators.

    This is the "old-style" deterministic fuzzer that generates data
    based purely on the OpenAPI schema without LLM involvement.

    Args:
        test_plan: The test plan from the preliminary phase
        openapi_spec: The OpenAPI specification
        credentials: Optional credentials to use for authentication steps
        seed: Optional seed for reproducible random generation

    Returns:
        GeneratedTestData containing mock payloads for each test step
    """
    parser = OpenAPISchemaParser(openapi_spec)
    generator = DeterministicGenerator(seed=seed)

    payloads: list[MockedPayload] = []

    for step in test_plan.steps:
        # Get request body schema and generate data
        request_body: dict[str, Any] | None = None
        body_schema = parser.get_request_body_schema(step.endpoint, step.method.lower())

        if body_schema:
            request_body = generate_data_from_schema(body_schema, parser, generator)

        # Handle auth provider steps - use credentials
        if step.auth_requirement == AuthRequirement.AUTH_PROVIDER and credentials:
            # Override with actual credentials for auth endpoints
            if request_body is None:
                request_body = {}
            if credentials.username:
                # Try to set username in various possible field names
                for field in ["username", "email", "user", "login"]:
                    if field in request_body or not request_body:
                        request_body[field] = credentials.username
                        break
            if credentials.password:
                request_body["password"] = credentials.password
            # Add grant_type for OAuth2 flows
            params = parser.get_parameters_schema(step.endpoint, step.method.lower())
            form_params = params.get("formData", [])
            for param in form_params:
                if param.get("name") == "grant_type":
                    request_body["grant_type"] = "password"

        # Generate path parameters
        path_params: dict[str, Any] = {}
        param_names = _extract_path_params(step.endpoint)
        params_schema = parser.get_parameters_schema(step.endpoint, step.method.lower())

        for param in params_schema.get("path", []):
            param_name = param.get("name", "")
            if param_name in param_names:
                param_schema = param.get("schema", param)  # Swagger 2.0 vs OpenAPI 3.x
                path_params[param_name] = generate_data_from_schema(
                    param_schema, parser, generator, param_name
                )

        # Generate query parameters (required only)
        query_params: dict[str, Any] = {}
        for param in params_schema.get("query", []):
            if param.get("required", False):
                param_name = param.get("name", "")
                param_schema = param.get("schema", param)
                query_params[param_name] = generate_data_from_schema(
                    param_schema, parser, generator, param_name
                )

        # Generate headers
        headers: dict[str, str] = {}

        # Add authorization header for protected endpoints
        if step.auth_requirement == AuthRequirement.REQUIRED:
            auth_step_id = _find_auth_provider_step(step.id, test_plan, step.depends_on)
            if auth_step_id:
                headers["Authorization"] = f"Bearer {{{{{auth_step_id}.access_token}}}}"

        # Add required header parameters
        for param in params_schema.get("header", []):
            if param.get("required", False):
                param_name = param.get("name", "")
                param_schema = param.get("schema", param)
                value = generate_data_from_schema(
                    param_schema, parser, generator, param_name
                )
                headers[param_name] = str(value)

        payloads.append(
            MockedPayload(
                step_id=step.id,
                request_body=request_body,
                path_params=path_params,
                query_params=query_params,
                headers=headers,
            )
        )

    return GeneratedTestData(payloads=payloads)


async def run_data_generation_phase(
    test_plan: APIPlan,
    openapi_spec: dict,
    credentials: APICredentials | None = None,
    use_llm: bool = True,
    per_step: bool = True,
    seed: int | str | None = None,
) -> GeneratedTestData:
    """
    Run the data generation phase: generate mock data for all test steps.

    Args:
        test_plan: The test plan from the preliminary phase
        openapi_spec: The OpenAPI specification
        credentials: Optional credentials to use for authentication steps
        use_llm: If True, use LLM-based generation; if False, use deterministic generation
        per_step: If True and use_llm=True, generate data per-step with weak model (default).
                  If False and use_llm=True, generate all data in one bulk request with strong model.
        seed: Optional seed for deterministic generation (only used when use_llm=False)

    Returns:
        GeneratedTestData containing mock payloads for each test step
    """
    if not use_llm:
        return run_deterministic_data_generation(
            test_plan=test_plan,
            openapi_spec=openapi_spec,
            credentials=credentials,
            seed=seed,
        )

    if per_step:
        # Per-step generation with weak model
        return await _run_per_step_generation(
            test_plan=test_plan,
            openapi_spec=openapi_spec,
            credentials=credentials,
        )

    # Bulk generation with strong model (legacy mode)
    return await _run_bulk_generation(
        test_plan=test_plan,
        openapi_spec=openapi_spec,
        credentials=credentials,
    )


async def _run_per_step_generation(
    test_plan: APIPlan,
    openapi_spec: dict,
    credentials: APICredentials | None = None,
) -> GeneratedTestData:
    """
    Generate test data one step at a time using the weak model.

    This approach is more reliable for large test plans because:
    1. Each request is small and focused
    2. Less likely to hit token limits or timeouts
    3. Easier to debug individual step failures
    4. Guarantees one payload per step (with fallback on failure)
    """
    payloads: list[MockedPayload] = []
    previous_step_ids: list[str] = []
    failed_step_ids: list[str] = []

    # Find the first auth provider step for use in subsequent steps
    auth_provider_step_id: str | None = None
    for step in test_plan.steps:
        if step.auth_requirement == AuthRequirement.AUTH_PROVIDER:
            auth_provider_step_id = step.id
            break

    total_steps = len(test_plan.steps)

    for i, step in enumerate(test_plan.steps, 1):
        logger.info("Generating data for step %d/%d: %s", i, total_steps, step.id)

        # Find the auth provider for this specific step (from its dependencies)
        step_auth_provider = _find_auth_provider_step(
            step.id, test_plan, step.depends_on
        )
        if step_auth_provider is None:
            step_auth_provider = auth_provider_step_id

        try:
            payload = await _generate_single_step_data(
                step=step,
                openapi_spec=openapi_spec,
                credentials=credentials,
                previous_step_ids=previous_step_ids.copy(),
                auth_provider_step_id=step_auth_provider,
            )
            payloads.append(payload)
        except Exception as e:
            logger.error("Failed to generate data for step %s: %s", step.id, e)
            failed_step_ids.append(step.id)
            # Create a minimal fallback payload
            payloads.append(
                MockedPayload(
                    step_id=step.id,
                    request_body=None,
                    path_params={},
                    query_params={},
                    headers={},
                )
            )

        previous_step_ids.append(step.id)

    # Log final status
    if failed_step_ids:
        logger.warning(
            "Failed to generate proper data for %d step(s) (using fallback): %s",
            len(failed_step_ids),
            failed_step_ids,
        )
    else:
        logger.info("Successfully generated data for all %d step(s)", total_steps)

    return GeneratedTestData(payloads=payloads)


async def _run_bulk_generation(
    test_plan: APIPlan,
    openapi_spec: dict,
    credentials: APICredentials | None = None,
) -> GeneratedTestData:
    """
    Generate test data for all steps in one bulk request using the strong model.

    This approach includes validation and retry logic to ensure all steps
    in the test plan have generated data.
    """
    deps = DataGenerationDeps(
        test_plan=test_plan,
        openapi_spec=openapi_spec,
        credentials=credentials,
    )

    prompt = """
    Generate mock data for all test steps in the test plan. 
    Use the tools to retrieve the test plan, OpenAPI specification, and test credentials.
    
    IMPORTANT: For authentication steps (auth_requirement="auth_provider"), you MUST use 
    the exact credentials returned by the get_test_credentials tool. Do not make up 
    different usernames or passwords.
    
    Ensure data consistency across dependent steps.
    """

    result = await data_generation_agent.run(prompt, deps=deps)
    current_data = result.output

    # Filter out any extra payloads not in the test plan
    extra_ids = find_extra_step_ids(test_plan, current_data)
    if extra_ids:
        logger.warning(
            "Removing %d extra payloads not in test plan: %s",
            len(extra_ids),
            sorted(extra_ids),
        )
        current_data = filter_valid_payloads(test_plan, current_data)

    # Check for missing steps and retry if needed
    for retry_num in range(1, MAX_DATA_GENERATION_RETRIES + 1):
        missing_ids = find_missing_step_ids(test_plan, current_data)

        if not missing_ids:
            logger.info("Full data coverage achieved after %d attempt(s)", retry_num)
            break

        logger.info(
            "Attempt %d: Found %d missing step(s), retrying...",
            retry_num,
            len(missing_ids),
        )

        # Build prompt for missing steps only
        missing_steps = get_missing_steps(test_plan, current_data)
        missing_list = "\n".join(
            f"  - {step.id}: {step.method} {step.endpoint}"
            for step in missing_steps
        )

        retry_prompt = f"""Generate mock data for the following steps that are missing from the previous generation.

Missing steps:
{missing_list}

Use the same format as before. Only generate data for these specific steps.

IMPORTANT: Use the exact step_id values listed above.
"""

        try:
            retry_result = await data_generation_agent.run(retry_prompt, deps=deps)
            additional_data = retry_result.output
            current_data = merge_test_data(current_data, additional_data, test_plan)
        except Exception as e:
            logger.warning("Retry %d failed: %s", retry_num, e)
            continue

    # Log final coverage status
    final_missing = find_missing_step_ids(test_plan, current_data)
    if final_missing:
        logger.warning(
            "Could not generate data for %d step(s): %s",
            len(final_missing),
            sorted(final_missing),
        )
    else:
        logger.info(
            "Generated data for all %d step(s) in test plan",
            len(test_plan.steps),
        )

    return current_data
