import asyncio
import json
import logfire
from pathlib import Path
import argparse
from devtools import debug
import re
from datetime import datetime
from dotenv import load_dotenv

from llama_wrestler.phases import (
    run_preliminary_phase,
    run_data_generation_phase,
    run_test_execution_phase,
)
from llama_wrestler.models import APICredentials
from llama_wrestler.spec_utils import (
    compute_spec_hash,
    save_spec_hash,
    find_cached_test_plan,
    sort_api_plan,
    validate_auth_requirements,
)

# Load environment variables from .env file
load_dotenv()

# Configure logfire
logfire.configure()


def get_next_run_id(output_dir: Path) -> int:
    if not output_dir.exists():
        return 1

    max_id = 0
    for item in output_dir.iterdir():
        try:
            # Expecting format: NNN_url_timestamp/
            parts = item.name.split("_")
            if parts and parts[0].isdigit():
                run_id = int(parts[0])
                if run_id > max_id:
                    max_id = run_id
        except Exception:
            continue
    return max_id + 1


def normalize_url(url: str) -> str:
    # Remove scheme
    url = re.sub(r"^https?://", "", url)
    # Replace non-alphanumeric characters with underscores
    url = re.sub(r"[^a-zA-Z0-9]", "_", url)
    # Remove multiple underscores
    url = re.sub(r"_+", "_", url)
    # Trim underscores from ends
    url = url.strip("_")
    return url[:50]  # Limit length


def create_run_directory(output_dir: Path, url: str) -> Path:
    # Generate filename
    run_id = get_next_run_id(output_dir)
    normalized_url = normalize_url(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = output_dir / f"{run_id:03d}_{normalized_url}_{timestamp}"
    run_directory.mkdir(exist_ok=True)
    return run_directory


async def run():
    parser = argparse.ArgumentParser(description="Generate and run REST API tests.")
    parser.add_argument("openapi_url", help="URL to the OpenAPI specification")
    parser.add_argument(
        "--repo", help="Path to the local repository for analysis", default=None
    )
    parser.add_argument(
        "--skip-execution", action="store_true", help="Skip the test execution phase"
    )
    # Credential arguments
    parser.add_argument(
        "--username", "-u", help="Username/email for authentication", default=None
    )
    parser.add_argument(
        "--password", "-p", help="Password for authentication", default=None
    )
    parser.add_argument(
        "--credentials-file",
        "-c",
        help="Path to JSON file with credentials (username, password, and extra fields)",
        default=None,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable test plan caching (always regenerate)",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo) if args.repo else None

    # Build credentials
    credentials = None
    if args.credentials_file:
        creds_path = Path(args.credentials_file)
        if creds_path.exists():
            with open(creds_path) as f:
                creds_data = json.load(f)
            credentials = APICredentials(**creds_data)
            print(f"Loaded credentials from {args.credentials_file}")
        else:
            print(f"Warning: Credentials file not found: {args.credentials_file}")
    elif args.username or args.password:
        credentials = APICredentials(username=args.username, password=args.password)
        print(f"Using provided credentials for user: {args.username}")

    # Output directory setup
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    run_directory = create_run_directory(output_dir, args.openapi_url)

    # ==================== Phase 1: Preliminary ====================
    print(f"\n{'=' * 60}")
    print("PHASE 1: Preliminary Analysis")
    print(f"{'=' * 60}")
    print(f"Analyzing OpenAPI spec at: {args.openapi_url}")
    if repo_path:
        print(f"Analyzing repository at: {repo_path}")

    # First, fetch the OpenAPI spec to check for cached test plans
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(args.openapi_url)
        response.raise_for_status()
        openapi_spec = response.json()

    # Compute hash of normalized spec
    spec_hash = compute_spec_hash(openapi_spec)
    print(f"Spec hash: {spec_hash[:16]}...")

    # Check for cached test plan
    cached_plan = None
    if not args.no_cache:
        cached_plan = find_cached_test_plan(output_dir, spec_hash)
        if cached_plan:
            print(f"Found cached test plan from previous run!")

    if cached_plan:
        # Use cached plan
        test_plan = cached_plan
        print("Using cached test plan (use --no-cache to regenerate)")
    else:
        # Generate new test plan
        preliminary_result = await run_preliminary_phase(
            openapi_url=args.openapi_url,
            repo_path=repo_path,
        )
        test_plan = preliminary_result.test_plan
        # Update openapi_spec from preliminary result if available
        if preliminary_result.openapi_spec:
            openapi_spec = preliminary_result.openapi_spec

    # Sort the test plan in topological + lexicographical order
    test_plan = sort_api_plan(test_plan)

    print("\n--- Generated Test Plan ---")
    debug(test_plan)

    # Validate auth requirements against spec
    auth_warnings = validate_auth_requirements(test_plan, openapi_spec)
    if auth_warnings:
        print("\n--- Authentication Validation Warnings ---")
        for warning in auth_warnings:
            print(f"  ⚠️  {warning}")

    # Save the OpenAPI spec
    openapi_filename = run_directory / "openapi_spec.json"
    with open(openapi_filename, "w") as f:
        json.dump(openapi_spec, f, indent=2)
    print(f"OpenAPI spec saved to {openapi_filename}")

    # Save the spec hash
    save_spec_hash(run_directory, spec_hash)
    print(f"Spec hash saved to {run_directory / 'spec_hash.txt'}")

    # Save the test plan (sorted)
    test_plan_filename = run_directory / "test_plan.json"
    with open(test_plan_filename, "w") as f:
        f.write(test_plan.model_dump_json(indent=2))
    print(f"Test plan saved to {test_plan_filename}")

    # ==================== Phase 2: Data Generation ====================
    print(f"\n{'=' * 60}")
    print("PHASE 2: Test Data Generation (Deterministic)")
    print(f"{'=' * 60}")

    test_data = await run_data_generation_phase(
        test_plan=test_plan,
        openapi_spec=openapi_spec,
        credentials=credentials,
    )

    print("\n--- Generated Test Data ---")
    debug(test_data)

    # Save the generated test data
    test_data_filename = run_directory / "test_data.json"
    with open(test_data_filename, "w") as f:
        f.write(test_data.model_dump_json(indent=2))
    print(f"Test data saved to {test_data_filename}")

    # ==================== Phase 3: Test Execution ====================
    if args.skip_execution:
        print(f"\n{'=' * 60}")
        print("PHASE 3: Test Execution (SKIPPED)")
        print(f"{'=' * 60}")
        return

    print(f"\n{'=' * 60}")
    print("PHASE 3: Test Execution")
    print(f"{'=' * 60}")

    execution_result = await run_test_execution_phase(
        test_plan=test_plan,
        test_data=test_data,
    )

    print("\n--- Test Execution Results ---")
    print(f"Total: {execution_result.total_steps}")
    print(f"Passed: {execution_result.passed}")
    print(f"Failed: {execution_result.failed}")
    print(f"Skipped: {execution_result.skipped}")

    # Print individual results
    for result in execution_result.results:
        status = "✓" if result.success else "✗"
        print(
            f"  {status} {result.step_id}: {result.status_code or 'N/A'} (expected {result.expected_status})"
        )
        if result.error:
            print(f"      Error: {result.error}")

    # Save the execution results
    execution_result_filename = run_directory / "execution_results.json"
    with open(execution_result_filename, "w") as f:
        f.write(execution_result.model_dump_json(indent=2))
    print(f"\nExecution results saved to {execution_result_filename}")


if __name__ == "__main__":
    asyncio.run(run())


def main():
    """Entry point for the console script."""
    asyncio.run(run())
