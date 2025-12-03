"""
Utilities for OpenAPI spec normalization, hashing, caching, and validation.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from llama_wrestler.models import APIPlan, APIStep, AuthRequirement


def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize an OpenAPI spec for consistent hashing.

    This function:
    - Recursively sorts all dictionary keys
    - Removes fields that don't affect the API structure (descriptions, examples, etc.)
    - Normalizes values for consistency

    Args:
        spec: The raw OpenAPI spec dictionary

    Returns:
        A normalized dictionary suitable for consistent hashing
    """
    # Fields to exclude from normalization (they don't affect API structure)
    EXCLUDE_FIELDS = {
        "description",
        "summary",
        "externalDocs",
        "termsOfService",
        "contact",
        "license",
        "x-logo",
        "x-tagGroups",
        "example",
        "examples",
        "deprecated",  # Keep deprecated status but it doesn't change behavior
    }

    def _normalize_value(value: Any, exclude_fields: set[str] | None = None) -> Any:
        """Recursively normalize a value."""
        if exclude_fields is None:
            exclude_fields = EXCLUDE_FIELDS

        if isinstance(value, dict):
            # Sort keys and recursively normalize values
            normalized = {}
            for key in sorted(value.keys()):
                # Skip excluded fields at any level
                if key in exclude_fields:
                    continue
                normalized[key] = _normalize_value(value[key], exclude_fields)
            return normalized
        elif isinstance(value, list):
            # Normalize each list item
            return [_normalize_value(item, exclude_fields) for item in value]
        elif isinstance(value, str):
            # Normalize strings (strip whitespace)
            return value.strip()
        else:
            return value

    return _normalize_value(spec)


def compute_spec_hash(spec: dict[str, Any]) -> str:
    """
    Compute a SHA-256 hash of a normalized OpenAPI spec.

    Args:
        spec: The OpenAPI spec (will be normalized first)

    Returns:
        A hex string of the SHA-256 hash
    """
    normalized = normalize_spec(spec)
    # Use json.dumps with sort_keys for deterministic serialization
    spec_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(spec_json.encode("utf-8")).hexdigest()


def save_spec_hash(run_dir: Path, spec_hash: str) -> None:
    """
    Save the spec hash to a file in the run directory.

    Args:
        run_dir: The run output directory
        spec_hash: The computed hash string
    """
    hash_file = run_dir / "spec_hash.txt"
    hash_file.write_text(spec_hash)


def load_spec_hash(run_dir: Path) -> str | None:
    """
    Load the spec hash from a run directory if it exists.

    Args:
        run_dir: The run output directory

    Returns:
        The hash string, or None if not found
    """
    hash_file = run_dir / "spec_hash.txt"
    if hash_file.exists():
        return hash_file.read_text().strip()
    return None


def find_cached_test_plan(output_dir: Path, spec_hash: str) -> APIPlan | None:
    """
    Look up a cached test plan from previous runs with the same spec hash.

    Scans all existing run directories for a matching spec hash and returns
    the test plan from the most recent matching run.

    Args:
        output_dir: The base output directory containing all runs
        spec_hash: The hash of the current normalized spec

    Returns:
        The cached APIPlan if found, None otherwise
    """
    if not output_dir.exists():
        return None

    matching_runs: list[tuple[Path, str]] = []

    # Scan all run directories
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        cached_hash = load_spec_hash(run_dir)
        if cached_hash == spec_hash:
            test_plan_file = run_dir / "test_plan.json"
            if test_plan_file.exists():
                # Extract timestamp from directory name for sorting
                # Format: NNN_url_timestamp
                dir_name = run_dir.name
                matching_runs.append((run_dir, dir_name))

    if not matching_runs:
        return None

    # Sort by directory name (which includes timestamp) to get the most recent
    matching_runs.sort(key=lambda x: x[1], reverse=True)

    # Load the test plan from the most recent matching run
    latest_run_dir = matching_runs[0][0]
    test_plan_file = latest_run_dir / "test_plan.json"

    try:
        with open(test_plan_file, "r") as f:
            plan_data = json.load(f)
        return APIPlan.model_validate(plan_data)
    except Exception:
        return None


def topological_sort_steps(steps: list[APIStep]) -> list[APIStep]:
    """
    Sort steps in topological order based on dependencies,
    with lexicographical ordering as a tiebreaker.

    Args:
        steps: List of API steps with dependencies

    Returns:
        A new list of steps in topological + lexicographical order
    """
    # Build dependency graph
    step_by_id: dict[str, APIStep] = {step.id: step for step in steps}
    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    # Initialize in-degrees
    for step in steps:
        if step.id not in in_degree:
            in_degree[step.id] = 0

    # Build graph edges
    for step in steps:
        for dep_id in step.depends_on:
            if dep_id in step_by_id:  # Only count valid dependencies
                in_degree[step.id] += 1
                dependents[dep_id].append(step.id)

    # Collect steps with no dependencies, sorted lexicographically
    available: list[str] = sorted(
        [step_id for step_id, degree in in_degree.items() if degree == 0]
    )

    result: list[APIStep] = []

    while available:
        # Take the lexicographically smallest available step
        current_id = available.pop(0)
        result.append(step_by_id[current_id])

        # Reduce in-degree for all dependents
        for dependent_id in dependents[current_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                # Insert in sorted position
                insert_pos = 0
                for i, avail_id in enumerate(available):
                    if dependent_id < avail_id:
                        insert_pos = i
                        break
                    insert_pos = i + 1
                available.insert(insert_pos, dependent_id)

    # If we couldn't process all steps, there's a cycle
    # Return what we have plus remaining steps in lexicographical order
    if len(result) < len(steps):
        processed_ids = {step.id for step in result}
        remaining = sorted(
            [step for step in steps if step.id not in processed_ids], key=lambda s: s.id
        )
        result.extend(remaining)

    return result


def sort_api_plan(plan: APIPlan) -> APIPlan:
    """
    Sort an API plan's steps in topological + lexicographical order.

    Args:
        plan: The API plan to sort

    Returns:
        A new APIPlan with sorted steps
    """
    sorted_steps = topological_sort_steps(plan.steps)
    return APIPlan(summary=plan.summary, base_url=plan.base_url, steps=sorted_steps)


def extract_security_requirements(spec: dict[str, Any]) -> dict[str, set[str]]:
    """
    Extract security requirements from an OpenAPI spec.

    Returns a mapping of endpoint (method + path) to required security schemes.

    Args:
        spec: The OpenAPI spec dictionary

    Returns:
        Dict mapping "METHOD /path" to set of security scheme names
    """
    requirements: dict[str, set[str]] = {}

    # Get global security definitions
    global_security: list[dict[str, Any]] = spec.get("security", [])

    # Determine spec version
    # is_openapi3 = spec.get("openapi", "").startswith("3")

    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
            operation = path_item.get(method)
            if operation is None or not isinstance(operation, dict):
                continue

            endpoint_key = f"{method.upper()} {path}"

            # Check for operation-level security
            op_security = operation.get("security")

            if op_security is not None:
                # Explicit security for this operation
                if op_security == []:
                    # Empty array means no auth required
                    requirements[endpoint_key] = set()
                else:
                    schemes = set()
                    for sec_req in op_security:
                        if isinstance(sec_req, dict):
                            schemes.update(sec_req.keys())
                    requirements[endpoint_key] = schemes
            elif global_security:
                # Fall back to global security
                schemes = set()
                for sec_req in global_security:
                    if isinstance(sec_req, dict):
                        schemes.update(sec_req.keys())
                requirements[endpoint_key] = schemes
            else:
                # No security requirement
                requirements[endpoint_key] = set()

    return requirements


def validate_auth_requirements(plan: APIPlan, spec: dict[str, Any]) -> list[str]:
    """
    Validate that the auth requirements in the test plan match the OpenAPI spec.

    Args:
        plan: The API test plan
        spec: The OpenAPI spec dictionary

    Returns:
        List of warning messages for mismatches
    """
    warnings: list[str] = []

    spec_requirements = extract_security_requirements(spec)

    for step in plan.steps:
        # Try to find a matching endpoint in the spec
        # Handle path parameters (e.g., /users/{id} vs /users/123)
        matching_spec_key = None
        for spec_key in spec_requirements:
            spec_method, spec_path = spec_key.split(" ", 1)
            if spec_method != step.method.upper():
                continue

            # Check for exact match
            if spec_path == step.endpoint:
                matching_spec_key = spec_key
                break

            # Check for path parameter match
            # Convert path params like {id} to regex pattern
            pattern = re.sub(r"\{[^}]+\}", r"[^/]+", spec_path)
            pattern = f"^{pattern}$"
            if re.match(pattern, step.endpoint):
                matching_spec_key = spec_key
                break

        if matching_spec_key is None:
            # Endpoint not in spec - can't validate
            continue

        required_schemes = spec_requirements[matching_spec_key]

        # Validate auth requirement
        if required_schemes:
            # Spec requires auth
            if step.auth_requirement == AuthRequirement.NONE:
                warnings.append(
                    f"Step '{step.id}' ({step.method} {step.endpoint}) "
                    f"has auth_requirement=NONE but spec requires: {', '.join(required_schemes)}"
                )
        else:
            # Spec doesn't require auth
            if step.auth_requirement == AuthRequirement.REQUIRED:
                warnings.append(
                    f"Step '{step.id}' ({step.method} {step.endpoint}) "
                    f"has auth_requirement=REQUIRED but spec has no security requirement"
                )

    return warnings


def suggest_auth_provider_steps(spec: dict[str, Any]) -> list[str]:
    """
    Identify endpoints that might be authentication providers based on spec patterns.

    Common patterns:
    - POST /auth/login, /login, /signin
    - POST /oauth/token, /token
    - POST /users/login, /api/login

    Args:
        spec: The OpenAPI spec dictionary

    Returns:
        List of endpoint keys (e.g., "POST /login") that might be auth providers
    """
    auth_patterns = [
        r".*/login$",
        r".*/signin$",
        r".*/authenticate$",
        r".*/auth$",
        r".*/token$",
        r".*/oauth/token$",
        r".*/oauth2/token$",
        r".*/api-token-auth$",
    ]

    auth_endpoints: list[str] = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # Check if path matches auth patterns
        path_lower = path.lower()
        is_auth_path = any(re.match(pattern, path_lower) for pattern in auth_patterns)

        if is_auth_path:
            # Auth endpoints are typically POST
            if "post" in path_item:
                auth_endpoints.append(f"POST {path}")

    return auth_endpoints
