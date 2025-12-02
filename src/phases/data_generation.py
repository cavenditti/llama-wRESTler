from dataclasses import dataclass
from typing import Any
import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from ..models import TestPlan
from ..settings import settings


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


data_generation_agent = Agent(
    model=f"openai:{settings.openai_model}",
    deps_type=DataGenerationDeps,
    output_type=GeneratedTestData,
    system_prompt="""
You are a test data generation expert. Your goal is to generate realistic mock data for API testing.

You will be provided with:
1. A test plan containing endpoint steps with their dependencies
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
- request_body: The JSON body for POST/PUT/PATCH requests, or null for GET/DELETE
- path_params: Dict of path parameters extracted from the endpoint URL template
- query_params: Dict of query parameters 
- headers: Dict of additional headers

IMPORTANT: You MUST fill in path_params and query_params based on the endpoint!

For endpoint "/pet/{petId}" -> path_params MUST contain {"petId": "value"}
For endpoint "/pet/findByStatus" with query -> query_params MUST contain {"status": "available"}
For endpoint "/user/login" with query -> query_params MUST contain {"username": "...", "password": "..."}

Examples:

1. GET /pet/{petId} - needs path_params:
{
  "step_id": "pet_get_by_id",
  "request_body": null,
  "path_params": {"petId": "{{pet_create.id}}"},
  "query_params": {},
  "headers": {}
}

2. GET /pet/findByStatus - needs query_params:
{
  "step_id": "pet_find_by_status_available",
  "request_body": null,
  "path_params": {},
  "query_params": {"status": "available"},
  "headers": {}
}

3. POST /pet - needs request_body only:
{
  "step_id": "pet_create",
  "request_body": {"name": "Rex", "photoUrls": ["http://example.com/img.jpg"], "status": "available"},
  "path_params": {},
  "query_params": {},
  "headers": {}
}

4. DELETE /pet/{petId} with api_key header:
{
  "step_id": "pet_delete",
  "request_body": null,
  "path_params": {"petId": "{{pet_create.id}}"},
  "query_params": {},
  "headers": {"api_key": "special-key"}
}

5. GET /user/login - needs query_params:
{
  "step_id": "user_login",
  "request_body": null,
  "path_params": {},
  "query_params": {"username": "{{user_create_single.username}}", "password": "{{user_create_single.password}}"},
  "headers": {}
}

6. GET /user/{username} - needs path_params:
{
  "step_id": "user_get_by_name",
  "request_body": null,
  "path_params": {"username": "{{user_create_single.username}}"},
  "query_params": {},
  "headers": {}
}

For dynamic values from previous steps, use placeholders like {{step_id.field_path}}.
Look at the endpoint template to determine which path parameters are needed.
Look at the OpenAPI spec to determine which query parameters are needed.
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


async def run_data_generation_phase(
    test_plan: TestPlan,
    openapi_spec: dict,
) -> GeneratedTestData:
    """
    Run the data generation phase: generate mock data for all test steps.
    
    Args:
        test_plan: The test plan from the preliminary phase
        openapi_spec: The OpenAPI specification
    
    Returns:
        GeneratedTestData containing mock payloads for each test step
    """
    deps = DataGenerationDeps(test_plan=test_plan, openapi_spec=openapi_spec)
    
    prompt = """
    Generate mock data for all test steps in the test plan. 
    Use the tools to retrieve the test plan and OpenAPI specification.
    Ensure data consistency across dependent steps.
    """
    
    result = await data_generation_agent.run(prompt, deps=deps)
    return result.output
