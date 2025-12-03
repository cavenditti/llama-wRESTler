"""
Data generation phase: Generate test data for API test steps.

This module provides deterministic data generation based on OpenAPI schemas,
with support for:
- Schema-based data generation
- Automatic placeholder generation for path params referencing dependencies
- Auth header placeholders for protected endpoints
- Credential injection for auth provider endpoints
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from llama_wrestler.models import (
    APIPlan,
    APIStep,
    APICredentials,
    AuthRequirement,
)
from llama_wrestler.schema import (
    OpenAPISchemaParser,
    DeterministicGenerator,
    generate_data_from_schema,
)

logger = logging.getLogger(__name__)


class MockedPayload(BaseModel):
    """Mocked data for a single test step."""

    step_id: str = Field(description="The ID of the test step this payload is for")
    request_body: dict | list | None = Field(
        None,
        description="The request body to send, if applicable (can be dict or list)",
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


def _infer_path_param_placeholder(
    param_name: str,
    step: APIStep,
    test_plan: APIPlan,
) -> str | None:
    """
    Infer a placeholder for a path parameter based on step dependencies.

    Uses heuristics to match path parameter names to likely source steps.
    For example:
    - {user_id} with depends_on=["create_user"] -> {{create_user.id}}
    - {id} with depends_on=["create_item"] -> {{create_item.id}}
    - {pet_id} with depends_on=["add_pet"] -> {{add_pet.id}}

    Args:
        param_name: The path parameter name (e.g., "user_id", "id")
        step: The current step
        test_plan: The full test plan

    Returns:
        A placeholder string like "{{step_id.id}}" or None if no match
    """
    if not step.depends_on:
        return None

    step_map = {s.id: s for s in test_plan.steps}
    param_lower = param_name.lower()

    # Common ID-like parameter patterns
    is_id_param = (
        param_lower in ("id", "Id", "ID")
        or param_lower.endswith("_id")
        or param_lower.endswith("id")
        or param_lower.endswith("Id")
    )

    if not is_id_param:
        return None

    # Extract the entity name from the parameter (e.g., "user" from "user_id")
    entity_name = None
    if param_lower.endswith("_id"):
        entity_name = param_lower[:-3]  # Remove "_id"
    elif param_lower.endswith("id") and len(param_lower) > 2:
        entity_name = param_lower[:-2]  # Remove "id"

    # Try to find a matching dependency step
    for dep_id in step.depends_on:
        dep = step_map.get(dep_id)
        if not dep:
            continue

        dep_id_lower = dep_id.lower()

        # Skip auth provider steps for non-auth params
        if dep.auth_requirement == AuthRequirement.AUTH_PROVIDER:
            continue

        # Match if:
        # 1. Param is just "id" and there's a creation-like dependency
        # 2. Entity name appears in the dependency step ID
        if param_lower == "id":
            # For generic "id", use the first non-auth dependency that looks like a creation
            if any(
                verb in dep_id_lower
                for verb in ("create", "add", "post", "new", "register")
            ):
                return f"{{{{{dep_id}.id}}}}"
        elif entity_name and entity_name in dep_id_lower:
            # Entity name matches dependency (e.g., "user" in "create_user")
            return f"{{{{{dep_id}.id}}}}"

    # Fallback: if there's exactly one non-auth dependency, use it
    non_auth_deps = [
        dep_id
        for dep_id in step.depends_on
        if step_map.get(dep_id)
        and step_map[dep_id].auth_requirement != AuthRequirement.AUTH_PROVIDER
    ]
    if len(non_auth_deps) == 1:
        return f"{{{{{non_auth_deps[0]}.id}}}}"

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


def find_missing_step_ids(test_plan: APIPlan, test_data: GeneratedTestData) -> set[str]:
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


def find_extra_step_ids(test_plan: APIPlan, test_data: GeneratedTestData) -> set[str]:
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
    valid_payloads = [p for p in test_data.payloads if p.step_id in valid_step_ids]
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


def get_missing_steps(
    test_plan: APIPlan, test_data: GeneratedTestData
) -> list[APIStep]:
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


# ============================================================================
# Deterministic data generation
# ============================================================================


def run_deterministic_data_generation(
    test_plan: APIPlan,
    openapi_spec: dict,
    credentials: APICredentials | None = None,
    seed: int | str | None = None,
) -> GeneratedTestData:
    """
    Generate test data deterministically using schema-based generators.

    This generates data based purely on the OpenAPI schema without LLM involvement.
    It handles:
    - Schema-based data generation from OpenAPI spec
    - Automatic placeholder generation for path params that reference dependencies
    - Auth header placeholders for protected endpoints
    - Credential injection for auth provider endpoints

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
        request_body: dict[str, Any] | list | None = None
        body_schema = parser.get_request_body_schema(step.endpoint, step.method.lower())

        if body_schema:
            request_body = generate_data_from_schema(body_schema, parser, generator)

        # Handle auth provider steps - use credentials
        if step.auth_requirement == AuthRequirement.AUTH_PROVIDER and credentials:
            # Override with actual credentials for auth endpoints
            if request_body is None:
                request_body = {}
            if isinstance(request_body, dict):
                if credentials.username:
                    # Try to set username in various possible field names
                    for field in ["username", "email", "user", "login"]:
                        if field in request_body or not request_body:
                            request_body[field] = credentials.username
                            break
                if credentials.password:
                    request_body["password"] = credentials.password
                # Add grant_type for OAuth2 flows
                params = parser.get_parameters_schema(
                    step.endpoint, step.method.lower()
                )
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
                # First, try to infer a placeholder from dependencies
                placeholder = _infer_path_param_placeholder(param_name, step, test_plan)
                if placeholder:
                    path_params[param_name] = placeholder
                else:
                    # Fall back to deterministic generation
                    param_schema = param.get(
                        "schema", param
                    )  # Swagger 2.0 vs OpenAPI 3.x
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


# ============================================================================
# Main entry point
# ============================================================================


async def run_data_generation_phase(
    test_plan: APIPlan,
    openapi_spec: dict,
    credentials: APICredentials | None = None,
    seed: int | str | None = None,
) -> GeneratedTestData:
    """
    Run the data generation phase: generate mock data for all test steps.

    Uses deterministic generation which is fast and handles:
    - Schema-based data generation from OpenAPI spec
    - Automatic placeholder generation for path params that reference dependencies
    - Auth header placeholders for protected endpoints
    - Credential injection for auth provider endpoints

    Args:
        test_plan: The test plan from the preliminary phase
        openapi_spec: The OpenAPI specification
        credentials: Optional credentials to use for authentication steps
        seed: Optional seed for deterministic generation

    Returns:
        GeneratedTestData containing mock payloads for each test step
    """
    return run_deterministic_data_generation(
        test_plan=test_plan,
        openapi_spec=openapi_spec,
        credentials=credentials,
        seed=seed,
    )
