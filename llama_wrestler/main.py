import asyncio
import json
import logfire
import logging
from pathlib import Path
import argparse
import re
from datetime import datetime
from dotenv import load_dotenv

from llama_wrestler.logging_config import (
    setup_logging,
    get_logger,
    DEFAULT_LEVEL,
)

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
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO level, -vv for DEBUG level)",
    )
    args = parser.parse_args()

    # Configure logging based on verbosity flags
    match args.verbose:
        case 0:
            log_level = DEFAULT_LEVEL  # UPDATE_LEVEL - show PHASE, STEP, UPDATE
        case 1:
            log_level = logging.INFO  # Show INFO and above
        case _:
            log_level = logging.DEBUG  # Show DEBUG and above

    setup_logging(level=log_level)
    logger = get_logger(__name__)

    repo_path = Path(args.repo) if args.repo else None

    # Build credentials
    credentials = None
    if args.credentials_file:
        creds_path = Path(args.credentials_file)
        if creds_path.exists():
            with open(creds_path) as f:
                creds_data = json.load(f)
            credentials = APICredentials(**creds_data)
            logger.info(f"Loaded credentials from {args.credentials_file}")
        else:
            logger.warning(f"Credentials file not found: {args.credentials_file}")
    elif args.username or args.password:
        credentials = APICredentials(username=args.username, password=args.password)
        logger.info(f"Using provided credentials for user: {args.username}")

    # Output directory setup
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    run_directory = create_run_directory(output_dir, args.openapi_url)

    # ==================== Phase 1: Preliminary ====================
    logger.phase("PHASE 1: Preliminary Analysis")
    logger.update(f"Analyzing OpenAPI spec at: {args.openapi_url}")
    if repo_path:
        logger.update(f"Analyzing repository at: {repo_path}")

    # First, fetch the OpenAPI spec to check for cached test plans
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(args.openapi_url)
        response.raise_for_status()
        openapi_spec = response.json()

    # Compute hash of normalized spec
    spec_hash = compute_spec_hash(openapi_spec)
    logger.update(f"Spec hash: {spec_hash[:16]}...")

    # Check for cached test plan
    cached_plan = None
    if not args.no_cache:
        cached_plan = find_cached_test_plan(output_dir, spec_hash)
        if cached_plan:
            logger.info("Found cached test plan from previous run!")

    if cached_plan:
        # Use cached plan
        test_plan = cached_plan
        logger.info("Using cached test plan (use --no-cache to regenerate)")
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

    logger.debugf(test_plan)

    # Validate auth requirements against spec
    auth_warnings = validate_auth_requirements(test_plan, openapi_spec)
    if auth_warnings:
        for warning in auth_warnings:
            logger.warning(f"  {warning}")

        # Auto-fix auth requirements based on OpenAPI spec
        logger.update("Auto-fixing auth requirements...")
        test_plan = fix_auth_requirements_from_spec(test_plan, openapi_spec)

        # Re-validate to confirm fixes
        remaining_warnings = validate_auth_requirements(test_plan, openapi_spec)
        if remaining_warnings:
            logger.warning("Some warnings could not be auto-fixed:")
            for warning in remaining_warnings:
                logger.warning(f"  {warning}")
        else:
            logger.update("✓ All auth requirements fixed!")

    # Save the OpenAPI spec
    openapi_filename = run_directory / "openapi_spec.json"
    with open(openapi_filename, "w") as f:
        json.dump(openapi_spec, f, indent=2)
    logger.update(f"OpenAPI spec saved to {openapi_filename}")

    # Save the spec hash
    save_spec_hash(run_directory, spec_hash)
    logger.update(f"Spec hash saved to {run_directory / 'spec_hash.txt'}")

    # Save the test plan (sorted)
    test_plan_filename = run_directory / "test_plan.json"
    with open(test_plan_filename, "w") as f:
        f.write(test_plan.model_dump_json(indent=2))
    logger.update(f"Test plan saved to {test_plan_filename}")

    # ==================== Phase 2: Data Generation ====================
    logger.phase("PHASE 2: Test Data Generation (Deterministic)")

    test_data = await run_data_generation_phase(
        test_plan=test_plan,
        openapi_spec=openapi_spec,
        credentials=credentials,
    )

    logger.debugf(test_data)

    # Save the generated test data
    test_data_filename = run_directory / "test_data.json"
    with open(test_data_filename, "w") as f:
        f.write(test_data.model_dump_json(indent=2))
    logger.update(f"Test data saved to {test_data_filename}")

    # ==================== Phase 3: Test Execution ====================
    if args.skip_execution:
        logger.phase("PHASE 3: Test Execution (SKIPPED)")
        return

    logger.phase("PHASE 3: Test Execution")

    execution_result = await run_test_execution_phase(
        test_plan=test_plan,
        test_data=test_data,
        output_dir=run_directory,
        iteration=None,  # Initial execution
    )

    def print_execution_summary(result, iteration: int | None = None):
        """Print a summary of execution results."""
        iter_str = f" (Iteration {iteration})" if iteration else ""
        logger.update(
            f"Test Execution Results{iter_str}: {result.passed}/{result.total_steps} passed ({calculate_pass_rate(result):.1f}%)"
        )
        logger.info(f"  Failed: {result.failed}, Skipped: {result.skipped}")

    print_execution_summary(execution_result)

    # Log individual results
    for result in execution_result.results:
        status = "✓" if result.success else "✗"
        log_level = logging.INFO if result.success else logging.WARNING
        logger.log(
            log_level,
            f"  {status} {result.step_id}: {result.status_code or 'N/A'} (expected {result.expected_status})",
        )
        if result.error:
            logger.warning(f"      Error: {result.error}")

    # Save initial execution results
    execution_result_filename = run_directory / "execution_results.json"
    with open(execution_result_filename, "w") as f:
        f.write(execution_result.model_dump_json(indent=2))
    logger.update(f"Execution results saved to {execution_result_filename}")

    # ==================== Phase 4: Iterative Refinement ====================
    if args.no_refinement:
        logger.phase("PHASE 4: Refinement (SKIPPED)")
        return

    current_pass_rate = calculate_pass_rate(execution_result)
    refinable_failures = get_refinable_failure_count(execution_result)

    # Check if refinement is needed
    if current_pass_rate >= args.target_pass_rate:
        logger.update(
            f"✓ Target pass rate ({args.target_pass_rate}%) already achieved!"
        )
        return

    if refinable_failures == 0:
        logger.update(
            "⚠ No refinable failures found (all failures are dependency-based)"
        )
        return

    logger.phase("PHASE 4: Iterative Refinement")
    logger.update(
        f"Current: {current_pass_rate:.1f}% → Target: {args.target_pass_rate}% (max {args.max_iterations} iterations, {refinable_failures} refinable failures)"
    )

    # Initialize iteration history for regression tracking
    history = IterationHistory()
    # Record the initial execution
    history.record_iteration(execution_result, test_data)

    iteration = 0
    while iteration < args.max_iterations:
        iteration += 1
        logger.step(f"Refinement Iteration {iteration}/{args.max_iterations}")

        # Multi-run execution and recap generation
        pre_analysis_recap = None
        if not args.no_multi_run and args.execution_runs > 1:
            logger.update(
                f"Running {args.execution_runs} execution passes for stable failure detection..."
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

            logger.update(
                f"Multi-run: {len(consistently_passing)} passing, {len(consistently_failing)} failing, {len(flaky_steps)} flaky"
            )

            if flaky_steps:
                logger.warning(
                    f"Flaky tests: {', '.join(flaky_steps[:5])}{'...' if len(flaky_steps) > 5 else ''}"
                )

            # Use consensus result for refinement
            if aggregated_result.consensus_result:
                execution_result = aggregated_result.consensus_result

            # Generate failure recaps using weak model
            if consistently_failing or flaky_steps:
                logger.update(
                    f"Generating failure analysis recaps (batch size: {args.recap_batch_size})..."
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

                logger.update(f"Pre-Analysis: {recap.overall_summary}")
                if recap.priority_fixes:
                    logger.info(f"  Priority fixes: {len(recap.priority_fixes)}")

        # Run refinement phase
        logger.update("Analyzing failures and generating refined payloads...")
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

        logger.update(
            f"Refinement: {len(refinement_result.refined_payloads)} payloads, {len(refinement_result.refined_steps)} steps refined"
        )
        logger.info(f"  {refinement_result.analysis_summary}")

        # Report unfixable steps
        if refinement_result.unfixable_steps:
            logger.warning(
                f"  Steps marked unfixable: {len(refinement_result.unfixable_steps)}"
            )
            for unfixable in refinement_result.unfixable_steps:
                logger.warning(
                    f"      - {unfixable.step_id} ({unfixable.category}): {unfixable.reason[:60]}..."
                )

        # Report steps with broken dependencies that need fixing
        if execution_result.steps_with_broken_deps:
            logger.warning(
                f"  Steps with invalid dependencies: {len(execution_result.steps_with_broken_deps)}"
            )
            for step_id, broken_deps in execution_result.steps_with_broken_deps.items():
                logger.warning(f"      - {step_id}: missing [{', '.join(broken_deps)}]")

        # Report regressions that were detected and reverted
        regressions = history.get_regressions()
        if regressions:
            logger.warning(f"  Regressions detected and reverted: {len(regressions)}")
            for step_id in regressions:
                logger.warning(f"      - {step_id}")

        # Report steps with multiple failed attempts
        multi_failure_steps = history.get_steps_with_multiple_failures(min_attempts=3)
        if multi_failure_steps:
            logger.info(
                f"  Steps with 3+ failed fix attempts: {len(multi_failure_steps)}"
            )
            for step_id in multi_failure_steps[:5]:
                attempt_count = len(history.get_failed_attempts(step_id))
                logger.info(f"      - {step_id}: {attempt_count} failed attempts")
            if len(multi_failure_steps) > 5:
                logger.info(f"      ... and {len(multi_failure_steps) - 5} more")

        if (
            not refinement_result.refined_payloads
            and not refinement_result.refined_steps
        ):
            logger.update("⚠ No refinements made - stopping iteration")
            break

        # Save refined test plan if steps were modified
        if refinement_result.refined_steps:
            refined_plan_filename = run_directory / f"test_plan_iter{iteration}.json"
            with open(refined_plan_filename, "w") as f:
                f.write(test_plan.model_dump_json(indent=2))
            logger.update(f"Refined test plan saved to {refined_plan_filename}")

        # Save refined test data
        refined_data_filename = run_directory / f"test_data_iter{iteration}.json"
        with open(refined_data_filename, "w") as f:
            f.write(test_data.model_dump_json(indent=2))
        logger.update(f"Refined test data saved to {refined_data_filename}")

        # Re-execute tests with refined data
        logger.update("Re-executing tests with refined payloads...")
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
            logger.update(f"✓ Target pass rate ({args.target_pass_rate}%) achieved!")
            break

        if refinable_failures == 0:
            logger.update("⚠ No more refinable failures - stopping iteration")
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

    logger.phase("FINAL RESULTS")
    logger.update(
        f"Completed {iteration} iterations: {current_pass_rate:.1f}% pass rate ({execution_result.passed}/{execution_result.total_steps})"
    )
    logger.update(f"Results: {final_results_filename}")
    logger.update(f"Test data: {final_data_filename}")
    logger.update(f"Test plan: {final_plan_filename}")


if __name__ == "__main__":
    asyncio.run(run())


def main():
    """Entry point for the console script."""
    asyncio.run(run())
