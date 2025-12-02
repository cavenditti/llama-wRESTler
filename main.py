import asyncio
import json
import logfire
from pathlib import Path
import argparse
from devtools import debug
import re
from datetime import datetime
from dotenv import load_dotenv

from src.phases import (
    run_preliminary_phase,
    run_data_generation_phase,
    run_test_execution_phase,
)
from src.models import TestCredentials

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


async def main():
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
    args = parser.parse_args()

    repo_path = Path(args.repo) if args.repo else None

    # Build credentials
    credentials = None
    if args.credentials_file:
        creds_path = Path(args.credentials_file)
        if creds_path.exists():
            with open(creds_path) as f:
                creds_data = json.load(f)
            credentials = TestCredentials(**creds_data)
            print(f"Loaded credentials from {args.credentials_file}")
        else:
            print(f"Warning: Credentials file not found: {args.credentials_file}")
    elif args.username or args.password:
        credentials = TestCredentials(username=args.username, password=args.password)
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

    preliminary_result = await run_preliminary_phase(
        openapi_url=args.openapi_url,
        repo_path=repo_path,
    )

    print("\n--- Generated Test Plan ---")
    debug(preliminary_result.test_plan)

    # Save the OpenAPI spec
    if preliminary_result.openapi_spec:
        openapi_filename = run_directory / "openapi_spec.json"
        with open(openapi_filename, "w") as f:
            json.dump(preliminary_result.openapi_spec, f, indent=2)
        print(f"OpenAPI spec saved to {openapi_filename}")

    # Save the test plan
    test_plan_filename = run_directory / "test_plan.json"
    with open(test_plan_filename, "w") as f:
        f.write(preliminary_result.test_plan.model_dump_json(indent=2))
    print(f"Test plan saved to {test_plan_filename}")

    # Check if we have the OpenAPI spec for the next phases
    if not preliminary_result.openapi_spec:
        print(
            "\nWarning: No OpenAPI spec was fetched. Cannot proceed with data generation."
        )
        return

    # ==================== Phase 2: Data Generation ====================
    print(f"\n{'=' * 60}")
    print("PHASE 2: Test Data Generation")
    print(f"{'=' * 60}")

    test_data = await run_data_generation_phase(
        test_plan=preliminary_result.test_plan,
        openapi_spec=preliminary_result.openapi_spec,
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
        test_plan=preliminary_result.test_plan,
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
    asyncio.run(main())
