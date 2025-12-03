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
    run_refinement_phase,
    get_refinable_failure_count,
    calculate_pass_rate,
    fix_auth_requirements_from_spec,
    IterationHistory,
    run_multiple_executions,
    generate_failure_recaps,
    format_recap_for_refinement,
    update_fix_results_from_execution,
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
    # Refinement options
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--target-pass-rate",
        type=float,
        default=90.0,
        help="Target pass rate percentage to stop refinement (default: 90.0)",
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable iterative refinement (run once only)",
    )
    parser.add_argument(
        "--execution-runs",
        type=int,
        default=3,
        help="Number of execution runs per iteration for stable failure detection (default: 3)",
    )
    parser.add_argument(
        "--no-multi-run",
        action="store_true",
        help="Disable multi-run execution (use single run per iteration)",
    )
    parser.add_argument(
        "--recap-batch-size",
        type=int,
        default=5,
        help="Number of endpoints per batch for failure recap generation (default: 5)",
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
            print("Found cached test plan from previous run!")

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

        # Auto-fix auth requirements based on OpenAPI spec
        print("\n--- Auto-fixing auth requirements ---")
        test_plan = fix_auth_requirements_from_spec(test_plan, openapi_spec)

        # Re-validate to confirm fixes
        remaining_warnings = validate_auth_requirements(test_plan, openapi_spec)
        if remaining_warnings:
            print("  Some warnings could not be auto-fixed:")
            for warning in remaining_warnings:
                print(f"    ⚠️  {warning}")
        else:
            print("  ✓ All auth requirements fixed!")

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
        output_dir=run_directory,
        iteration=None,  # Initial execution
    )

    def print_execution_summary(result, iteration: int | None = None):
        """Print a summary of execution results."""
        iter_str = f" (Iteration {iteration})" if iteration else ""
        print(f"\n--- Test Execution Results{iter_str} ---")
        print(f"Total: {result.total_steps}")
        print(f"Passed: {result.passed}")
        print(f"Failed: {result.failed}")
        print(f"Skipped: {result.skipped}")
        print(f"Pass Rate: {calculate_pass_rate(result):.1f}%")

    print_execution_summary(execution_result)

    # Print individual results
    for result in execution_result.results:
        status = "✓" if result.success else "✗"
        print(
            f"  {status} {result.step_id}: {result.status_code or 'N/A'} (expected {result.expected_status})"
        )
        if result.error:
            print(f"      Error: {result.error}")

    # Save initial execution results
    execution_result_filename = run_directory / "execution_results.json"
    with open(execution_result_filename, "w") as f:
        f.write(execution_result.model_dump_json(indent=2))
    print(f"\nExecution results saved to {execution_result_filename}")

    # ==================== Phase 4: Iterative Refinement ====================
    if args.no_refinement:
        print(f"\n{'=' * 60}")
        print("PHASE 4: Refinement (SKIPPED)")
        print(f"{'=' * 60}")
        return

    current_pass_rate = calculate_pass_rate(execution_result)
    refinable_failures = get_refinable_failure_count(execution_result)

    # Check if refinement is needed
    if current_pass_rate >= args.target_pass_rate:
        print(f"\n✓ Target pass rate ({args.target_pass_rate}%) already achieved!")
        return

    if refinable_failures == 0:
        print("\n⚠ No refinable failures found (all failures are dependency-based)")
        return

    print(f"\n{'=' * 60}")
    print("PHASE 4: Iterative Refinement")
    print(f"{'=' * 60}")
    print(f"Current pass rate: {current_pass_rate:.1f}%")
    print(f"Target pass rate: {args.target_pass_rate}%")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Refinable failures: {refinable_failures}")
    if not args.no_multi_run:
        print(f"Execution runs per iteration: {args.execution_runs}")
        print(f"Recap batch size: {args.recap_batch_size}")

    # Initialize iteration history for regression tracking
    history = IterationHistory()
    # Record the initial execution
    history.record_iteration(execution_result, test_data)

    iteration = 0
    while iteration < args.max_iterations:
        iteration += 1
        print(f"\n{'─' * 40}")
        print(f"Refinement Iteration {iteration}/{args.max_iterations}")
        print(f"{'─' * 40}")

        # Multi-run execution and recap generation
        pre_analysis_recap = None
        if not args.no_multi_run and args.execution_runs > 1:
            print(
                f"\nRunning {args.execution_runs} execution passes for stable failure detection..."
            )
            aggregated_result = await run_multiple_executions(
                test_plan=test_plan,
                test_data=test_data,
                num_runs=args.execution_runs,
                openapi_spec=openapi_spec,
            )

            # Report aggregated statistics
            consistently_failing = aggregated_result.get_consistently_failing_steps()
            flaky_steps = aggregated_result.get_flaky_steps()
            consistently_passing = aggregated_result.get_consistently_passing_steps()

            print(f"\nMulti-run Analysis ({args.execution_runs} runs):")
            print(f"  Consistently passing: {len(consistently_passing)}")
            print(f"  Consistently failing: {len(consistently_failing)}")
            print(f"  Flaky (inconsistent): {len(flaky_steps)}")

            if flaky_steps:
                print(f"  ⚠️ Flaky tests detected: {', '.join(flaky_steps[:5])}")
                if len(flaky_steps) > 5:
                    print(f"      ... and {len(flaky_steps) - 5} more")

            # Use consensus result for refinement
            if aggregated_result.consensus_result:
                execution_result = aggregated_result.consensus_result

            # Generate failure recaps using weak model
            if consistently_failing or flaky_steps:
                print(
                    f"\nGenerating failure analysis recaps (batch size: {args.recap_batch_size})..."
                )
                recap = await generate_failure_recaps(
                    test_plan=test_plan,
                    aggregated_result=aggregated_result,
                    test_data=test_data,
                    openapi_spec=openapi_spec,
                    batch_size=args.recap_batch_size,
                    output_dir=run_directory,
                    iteration=iteration,
                )
                pre_analysis_recap = format_recap_for_refinement(recap)

                print("\nPre-Analysis Summary:")
                print(f"  {recap.overall_summary}")
                if recap.priority_fixes:
                    print(f"  Priority fixes identified: {len(recap.priority_fixes)}")
                    for fix in recap.priority_fixes[:3]:
                        print(f"    - {fix[:80]}...")

        # Run refinement phase
        print("\nAnalyzing failures and generating refined payloads...")
        test_plan, test_data, refinement_result = await run_refinement_phase(
            test_plan=test_plan,
            openapi_spec=openapi_spec,
            execution_result=execution_result,
            test_data=test_data,
            history=history,
            pre_analysis_recap=pre_analysis_recap,
            output_dir=run_directory,
            iteration=iteration,
        )

        print("\nRefinement Analysis:")
        print(f"  {refinement_result.analysis_summary}")
        print(f"  Payloads refined: {len(refinement_result.refined_payloads)}")
        print(f"  Steps refined: {len(refinement_result.refined_steps)}")

        # Report unfixable steps
        if refinement_result.unfixable_steps:
            print(
                f"  ⛔ Steps marked unfixable: {len(refinement_result.unfixable_steps)}"
            )
            for unfixable in refinement_result.unfixable_steps:
                print(
                    f"      - {unfixable.step_id} ({unfixable.category}): {unfixable.reason[:60]}..."
                )

        # Report steps with broken dependencies that need fixing
        if execution_result.steps_with_broken_deps:
            print(
                f"  ⚠️ Steps with invalid dependencies: {len(execution_result.steps_with_broken_deps)}"
            )
            for step_id, broken_deps in execution_result.steps_with_broken_deps.items():
                print(f"      - {step_id}: missing [{', '.join(broken_deps)}]")

        # Report regressions that were detected and reverted
        regressions = history.get_regressions()
        if regressions:
            print(f"  ⚠️ Regressions detected and reverted: {len(regressions)}")
            for step_id in regressions:
                print(f"      - {step_id}")

        # Report steps with multiple failed attempts
        multi_failure_steps = history.get_steps_with_multiple_failures(min_attempts=3)
        if multi_failure_steps:
            print(f"  📊 Steps with 3+ failed fix attempts: {len(multi_failure_steps)}")
            for step_id in multi_failure_steps[:5]:
                attempt_count = len(history.get_failed_attempts(step_id))
                print(f"      - {step_id}: {attempt_count} failed attempts")
            if len(multi_failure_steps) > 5:
                print(f"      ... and {len(multi_failure_steps) - 5} more")

        if (
            not refinement_result.refined_payloads
            and not refinement_result.refined_steps
        ):
            print("\n⚠ No refinements made - stopping iteration")
            break

        # Save refined test plan if steps were modified
        if refinement_result.refined_steps:
            refined_plan_filename = run_directory / f"test_plan_iter{iteration}.json"
            with open(refined_plan_filename, "w") as f:
                f.write(test_plan.model_dump_json(indent=2))
            print(f"Refined test plan saved to {refined_plan_filename}")

        # Save refined test data
        refined_data_filename = run_directory / f"test_data_iter{iteration}.json"
        with open(refined_data_filename, "w") as f:
            f.write(test_data.model_dump_json(indent=2))
        print(f"Refined test data saved to {refined_data_filename}")

        # Re-execute tests with refined data
        print("\nRe-executing tests with refined payloads...")
        execution_result = await run_test_execution_phase(
            test_plan=test_plan,
            test_data=test_data,
            output_dir=run_directory,
            iteration=iteration,
        )

        # Update fix results in history so we know what worked and what didn't
        update_fix_results_from_execution(history, execution_result)

        print_execution_summary(execution_result, iteration)

        # Save iteration results
        iter_results_filename = (
            run_directory / f"execution_results_iter{iteration}.json"
        )
        with open(iter_results_filename, "w") as f:
            f.write(execution_result.model_dump_json(indent=2))

        # Check if we've reached the target
        current_pass_rate = calculate_pass_rate(execution_result)
        refinable_failures = get_refinable_failure_count(execution_result)

        if current_pass_rate >= args.target_pass_rate:
            print(f"\n✓ Target pass rate ({args.target_pass_rate}%) achieved!")
            break

        if refinable_failures == 0:
            print("\n⚠ No more refinable failures - stopping iteration")
            break

    # Save final results
    final_results_filename = run_directory / "execution_results_final.json"
    with open(final_results_filename, "w") as f:
        f.write(execution_result.model_dump_json(indent=2))

    final_data_filename = run_directory / "test_data_final.json"
    with open(final_data_filename, "w") as f:
        f.write(test_data.model_dump_json(indent=2))

    final_plan_filename = run_directory / "test_plan_final.json"
    with open(final_plan_filename, "w") as f:
        f.write(test_plan.model_dump_json(indent=2))

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Iterations completed: {iteration}")
    print(f"Final pass rate: {current_pass_rate:.1f}%")
    print(f"Total passed: {execution_result.passed}/{execution_result.total_steps}")
    print(f"\nFinal results saved to {final_results_filename}")
    print(f"Final test data saved to {final_data_filename}")
    print(f"Final test plan saved to {final_plan_filename}")


if __name__ == "__main__":
    asyncio.run(run())


def main():
    """Entry point for the console script."""
    asyncio.run(run())
