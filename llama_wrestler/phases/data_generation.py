from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from llama_wrestler.models import TestPlan, TestCredentials
from llama_wrestler.settings import settings


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
    """Dependencies for the data generation agent."""

    test_plan: TestPlan
    openapi_spec: dict
    credentials: TestCredentials | None = None


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
    import json

    return json.dumps(ctx.deps.openapi_spec, indent=2)


@data_generation_agent.tool
async def get_test_credentials(ctx: RunContext[DataGenerationDeps]) -> str:
    """Get the credentials to use for authentication steps. Use these EXACT credentials for auth_provider steps."""
    if ctx.deps.credentials:
        return ctx.deps.credentials.to_prompt_context()
    return "No credentials provided - generate realistic test credentials"


async def run_data_generation_phase(
    test_plan: TestPlan,
    openapi_spec: dict,
    credentials: TestCredentials | None = None,
) -> GeneratedTestData:
    """
    Run the data generation phase: generate mock data for all test steps.

    Args:
        test_plan: The test plan from the preliminary phase
        openapi_spec: The OpenAPI specification
        credentials: Optional credentials to use for authentication steps

    Returns:
        GeneratedTestData containing mock payloads for each test step
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
    return result.output
