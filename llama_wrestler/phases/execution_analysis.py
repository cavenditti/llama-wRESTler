"""
Execution analysis: Multi-run execution and failure recap generation.

This module handles:
1. Running test execution multiple times to gather stable failure data
2. Aggregating results across runs to identify consistent vs flaky failures
3. Generating batched recaps of failures using a weak model to help the refinement agent
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from llama_wrestler.models import APIPlan, APIStep
from llama_wrestler.phases.data_generation import GeneratedTestData
from llama_wrestler.phases.test_execution import (
    APIExecutionResult,
    StepResult,
    run_test_execution_phase,
)
from llama_wrestler.settings import settings

logger = logging.getLogger(__name__)


class StepExecutionStats(BaseModel):
    """Aggregated statistics for a single step across multiple runs."""
    
    step_id: str
    total_runs: int
    success_count: int
    failure_count: int
    skip_count: int
    
    # Track unique error messages and their frequencies
    error_frequencies: dict[str, int] = Field(default_factory=dict)
    
    # Track status codes seen
    status_code_frequencies: dict[int, int] = Field(default_factory=dict)
    
    # Sample responses (keep one successful and one failed response for context)
    sample_success_response: Any = None
    sample_failure_response: Any = None
    sample_failure_error: str | None = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.success_count / self.total_runs) * 100
    
    @property
    def is_consistently_failing(self) -> bool:
        """Returns True if step failed in all runs."""
        return self.failure_count == self.total_runs
    
    @property
    def is_consistently_passing(self) -> bool:
        """Returns True if step passed in all runs."""
        return self.success_count == self.total_runs
    
    @property
    def is_flaky(self) -> bool:
        """Returns True if step has mixed results."""
        return not self.is_consistently_failing and not self.is_consistently_passing


class AggregatedExecutionResult(BaseModel):
    """Aggregated results from multiple execution runs."""
    
    num_runs: int
    step_stats: dict[str, StepExecutionStats] = Field(default_factory=dict)
    
    # Store all individual results for detailed analysis
    individual_results: list[APIExecutionResult] = Field(default_factory=list)
    
    # Merged result representing the "consensus" outcome
    # Uses most common outcome for each step
    consensus_result: APIExecutionResult | None = None
    
    def get_consistently_failing_steps(self) -> list[str]:
        """Get step IDs that failed in every run."""
        return [
            step_id for step_id, stats in self.step_stats.items()
            if stats.is_consistently_failing
        ]
    
    def get_flaky_steps(self) -> list[str]:
        """Get step IDs with inconsistent results."""
        return [
            step_id for step_id, stats in self.step_stats.items()
            if stats.is_flaky
        ]
    
    def get_consistently_passing_steps(self) -> list[str]:
        """Get step IDs that passed in every run."""
        return [
            step_id for step_id, stats in self.step_stats.items()
            if stats.is_consistently_passing
        ]


class EndpointRecap(BaseModel):
    """Recap of issues for a single endpoint."""
    
    step_id: str
    endpoint: str
    method: str
    issue_summary: str = Field(description="Brief summary of the main issue")
    likely_cause: str = Field(description="Most likely root cause")
    suggested_fix: str = Field(description="Suggested fix approach")
    confidence: str = Field(description="Confidence level: high, medium, low")


class BatchRecap(BaseModel):
    """Recap of a batch of endpoints analyzed together."""
    
    batch_summary: str = Field(description="Overall summary of issues in this batch")
    endpoint_recaps: list[EndpointRecap] = Field(default_factory=list)
    common_patterns: list[str] = Field(
        default_factory=list,
        description="Common patterns observed across endpoints"
    )


class FullAnalysisRecap(BaseModel):
    """Complete analysis recap across all batches."""
    
    overall_summary: str = Field(description="High-level summary of all issues")
    batch_recaps: list[BatchRecap] = Field(default_factory=list)
    priority_fixes: list[str] = Field(
        default_factory=list,
        description="Prioritized list of fixes to attempt"
    )
    flaky_step_notes: str | None = Field(
        None,
        description="Notes about flaky tests if any were detected"
    )


async def run_multiple_executions(
    test_plan: APIPlan,
    test_data: GeneratedTestData,
    num_runs: int = 3,
    openapi_spec: dict[str, Any] | None = None,
) -> AggregatedExecutionResult:
    """
    Run test execution multiple times and aggregate results.
    
    Args:
        test_plan: The test plan to execute
        test_data: The test data to use
        num_runs: Number of execution runs (default: 3)
        openapi_spec: Optional OpenAPI spec for validation
        
    Returns:
        AggregatedExecutionResult with statistics across all runs
    """
    logger.info("Running %d execution passes for stable failure detection", num_runs)
    
    individual_results: list[APIExecutionResult] = []
    step_stats: dict[str, StepExecutionStats] = {}
    
    # Initialize stats for all steps
    for step in test_plan.steps:
        step_stats[step.id] = StepExecutionStats(
            step_id=step.id,
            total_runs=0,
            success_count=0,
            failure_count=0,
            skip_count=0,
        )
    
    # Run executions
    for run_idx in range(num_runs):
        logger.info("Execution run %d/%d", run_idx + 1, num_runs)
        
        result = await run_test_execution_phase(
            test_plan=test_plan,
            test_data=test_data,
            openapi_spec=openapi_spec,
        )
        individual_results.append(result)
        
        # Aggregate results
        for step_result in result.results:
            stats = step_stats[step_result.step_id]
            stats.total_runs += 1
            
            if step_result.success:
                stats.success_count += 1
                if stats.sample_success_response is None:
                    stats.sample_success_response = step_result.response_body
            elif step_result.error == "Skipped due to failed dependencies":
                stats.skip_count += 1
            else:
                stats.failure_count += 1
                
                # Track error frequencies
                error_key = step_result.error or "Unknown error"
                stats.error_frequencies[error_key] = stats.error_frequencies.get(error_key, 0) + 1
                
                # Track status code frequencies
                if step_result.status_code is not None:
                    stats.status_code_frequencies[step_result.status_code] = \
                        stats.status_code_frequencies.get(step_result.status_code, 0) + 1
                
                # Keep sample failure
                if stats.sample_failure_response is None:
                    stats.sample_failure_response = step_result.response_body
                    stats.sample_failure_error = step_result.error
    
    # Build consensus result (use most common outcome for each step)
    consensus_result = _build_consensus_result(test_plan, step_stats, individual_results)
    
    # Copy broken deps from any run (should be consistent)
    if individual_results:
        consensus_result.steps_with_broken_deps = individual_results[0].steps_with_broken_deps
    
    return AggregatedExecutionResult(
        num_runs=num_runs,
        step_stats=step_stats,
        individual_results=individual_results,
        consensus_result=consensus_result,
    )


def _build_consensus_result(
    test_plan: APIPlan,
    step_stats: dict[str, StepExecutionStats],
    individual_results: list[APIExecutionResult],
) -> APIExecutionResult:
    """Build a consensus result using majority voting."""
    results: list[StepResult] = []
    passed = 0
    failed = 0
    skipped = 0
    
    # Create a map of step_id -> list of results across runs
    result_map: dict[str, list[StepResult]] = {}
    for exec_result in individual_results:
        for step_result in exec_result.results:
            if step_result.step_id not in result_map:
                result_map[step_result.step_id] = []
            result_map[step_result.step_id].append(step_result)
    
    for step in test_plan.steps:
        stats = step_stats[step.id]
        step_results = result_map.get(step.id, [])
        
        if not step_results:
            continue
        
        # Determine consensus outcome
        if stats.success_count > stats.failure_count and stats.success_count > stats.skip_count:
            # Majority success - use a successful result
            consensus = next((r for r in step_results if r.success), step_results[0])
            passed += 1
        elif stats.skip_count > stats.success_count and stats.skip_count > stats.failure_count:
            # Majority skipped
            consensus = next(
                (r for r in step_results if r.error == "Skipped due to failed dependencies"),
                step_results[0]
            )
            skipped += 1
        else:
            # Majority failure (or tie goes to failure for safety)
            consensus = next((r for r in step_results if not r.success), step_results[0])
            failed += 1
        
        results.append(consensus)
    
    return APIExecutionResult(
        total_steps=len(test_plan.steps),
        passed=passed,
        failed=failed,
        skipped=skipped,
        results=results,
    )


RECAP_SYSTEM_PROMPT = """
You are an API testing analyst. Your job is to analyze batches of failed API test results 
and create concise, actionable recaps that will help another AI fix the issues.

For each endpoint, you should:
1. Identify the core issue from the error response and status code
2. Determine the most likely root cause
3. Suggest a specific fix approach

Be concise but precise. Focus on actionable insights.

Common patterns to look for:
- Authentication issues (401, missing tokens, wrong auth scheme)
- Validation errors (400, 422 - missing fields, wrong types, invalid values)
- Resource not found (404 - wrong IDs, missing dependencies)
- Server errors (500 - may indicate payload schema issues)
- Flaky tests (inconsistent results across runs - note these specially)

When you see patterns across multiple endpoints, highlight them as they may have a common fix.
"""


@dataclass
class RecapDeps:
    """Dependencies for the recap agent."""
    test_plan: APIPlan
    openapi_spec: dict


def _create_recap_agent() -> Agent[RecapDeps, BatchRecap]:
    """Create the recap agent using the weak model."""
    return Agent(
        model=f"openai:{settings.openai_weak_model}",
        deps_type=RecapDeps,
        output_type=BatchRecap,
        system_prompt=RECAP_SYSTEM_PROMPT,
    )


async def generate_failure_recaps(
    test_plan: APIPlan,
    aggregated_result: AggregatedExecutionResult,
    test_data: GeneratedTestData,
    openapi_spec: dict[str, Any],
    batch_size: int = 5,
    output_dir: Path | None = None,
    iteration: int | None = None,
) -> FullAnalysisRecap:
    """
    Generate structured recaps of failures using the weak model.
    
    Processes failures in batches to handle large numbers of endpoints efficiently.
    
    Args:
        test_plan: The test plan
        aggregated_result: Results from multiple execution runs
        test_data: The test data used
        openapi_spec: The OpenAPI specification
        batch_size: Number of endpoints to analyze per batch
        output_dir: Optional directory to save request/response for analysis
        iteration: Optional iteration number for file naming
        
    Returns:
        FullAnalysisRecap with structured analysis
    """
    # Get failing steps (both consistent and flaky failures)
    consistently_failing = aggregated_result.get_consistently_failing_steps()
    flaky_steps = aggregated_result.get_flaky_steps()
    
    # For flaky steps, only include those that fail more than they pass
    failing_flaky = [
        step_id for step_id in flaky_steps
        if aggregated_result.step_stats[step_id].failure_count > 
           aggregated_result.step_stats[step_id].success_count
    ]
    
    all_failing = consistently_failing + failing_flaky
    
    if not all_failing:
        return FullAnalysisRecap(
            overall_summary="No failures detected across execution runs.",
            batch_recaps=[],
            priority_fixes=[],
            flaky_step_notes=None,
        )
    
    logger.info(
        "Generating recaps for %d failing steps (%d consistent, %d flaky)",
        len(all_failing), len(consistently_failing), len(failing_flaky)
    )
    
    # Build step map for lookups
    step_map = {s.id: s for s in test_plan.steps}
    payload_map = {p.step_id: p for p in test_data.payloads}
    
    # Process in batches
    batch_recaps: list[BatchRecap] = []
    agent = _create_recap_agent()
    deps = RecapDeps(test_plan=test_plan, openapi_spec=openapi_spec)
    
    for i in range(0, len(all_failing), batch_size):
        batch_step_ids = all_failing[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(all_failing) + batch_size - 1) // batch_size
        
        logger.info("Processing batch %d/%d (%d endpoints)", batch_num, total_batches, len(batch_step_ids))
        
        # Build context for this batch
        batch_context = _build_batch_context(
            batch_step_ids, step_map, aggregated_result, payload_map, openapi_spec
        )
        
        prompt = f"""Analyze this batch of {len(batch_step_ids)} failing API endpoints and create a recap.

{batch_context}

Provide:
1. A brief batch summary
2. For each endpoint: issue summary, likely cause, suggested fix, and confidence level
3. Any common patterns you notice across these endpoints
"""
        
        # Save request for analysis
        if output_dir:
            iter_suffix = f"_iter{iteration}" if iteration else ""
            timestamp = datetime.now().strftime("%H%M%S")
            request_file = output_dir / f"recap_batch{batch_num}_request{iter_suffix}_{timestamp}.md"
            with open(request_file, "w") as f:
                f.write(f"# Recap Batch {batch_num} Request\n")
                f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
                f.write(f"## System Prompt\n```\n{RECAP_SYSTEM_PROMPT}\n```\n\n")
                f.write(f"## User Prompt\n```\n{prompt}\n```\n")
            logger.info("Saved recap batch %d request to %s", batch_num, request_file)
        
        try:
            result = await agent.run(prompt, deps=deps)
            batch_recaps.append(result.output)
            
            # Save response for analysis
            if output_dir:
                response_file = output_dir / f"recap_batch{batch_num}_response{iter_suffix}_{timestamp}.json"
                with open(response_file, "w") as f:
                    json.dump(result.output.model_dump(), f, indent=2)
                logger.info("Saved recap batch %d response to %s", batch_num, response_file)
                
        except Exception as e:
            logger.error("Failed to generate recap for batch %d: %s", batch_num, e)
            # Create a fallback recap
            batch_recaps.append(BatchRecap(
                batch_summary=f"Failed to analyze batch: {e}",
                endpoint_recaps=[],
                common_patterns=[],
            ))
    
    # Generate overall summary
    overall_summary = _build_overall_summary(
        aggregated_result, consistently_failing, flaky_steps, batch_recaps
    )
    
    # Extract priority fixes from recaps
    priority_fixes = _extract_priority_fixes(batch_recaps)
    
    # Note about flaky tests
    flaky_notes = None
    if flaky_steps:
        flaky_notes = (
            f"Detected {len(flaky_steps)} flaky tests with inconsistent results: "
            f"{', '.join(flaky_steps[:5])}"
            f"{' and more...' if len(flaky_steps) > 5 else ''}"
        )
    
    # Save full recap for analysis
    full_recap = FullAnalysisRecap(
        overall_summary=overall_summary,
        batch_recaps=batch_recaps,
        priority_fixes=priority_fixes,
        flaky_step_notes=flaky_notes,
    )
    
    if output_dir:
        iter_suffix = f"_iter{iteration}" if iteration else ""
        recap_file = output_dir / f"full_recap{iter_suffix}.json"
        with open(recap_file, "w") as f:
            json.dump(full_recap.model_dump(), f, indent=2)
        logger.info("Saved full recap to %s", recap_file)
    
    return full_recap


def _build_batch_context(
    step_ids: list[str],
    step_map: dict[str, APIStep],
    aggregated_result: AggregatedExecutionResult,
    payload_map: dict[str, Any],
    openapi_spec: dict[str, Any],
) -> str:
    """Build context string for a batch of failing endpoints."""
    parts = []
    paths = openapi_spec.get("paths", {})
    
    for step_id in step_ids:
        step = step_map.get(step_id)
        stats = aggregated_result.step_stats.get(step_id)
        payload = payload_map.get(step_id)
        
        if not step or not stats:
            continue
        
        parts.append(f"\n## {step_id}")
        parts.append(f"- Endpoint: {step.method} {step.endpoint}")
        parts.append(f"- Description: {step.description}")
        parts.append(f"- Success Rate: {stats.success_rate:.0f}% ({stats.success_count}/{stats.total_runs})")
        
        if stats.is_flaky:
            parts.append("- ⚠️ FLAKY TEST - inconsistent results across runs")
        
        # Most common error
        if stats.error_frequencies:
            most_common_error = max(stats.error_frequencies.items(), key=lambda x: x[1])
            parts.append(f"- Most Common Error: {most_common_error[0]} (seen {most_common_error[1]}x)")
        
        # Status codes seen
        if stats.status_code_frequencies:
            status_str = ", ".join(f"{code}({count}x)" for code, count in stats.status_code_frequencies.items())
            parts.append(f"- Status Codes: {status_str}")
        
        # Sample failure response
        if stats.sample_failure_response:
            try:
                resp_str = json.dumps(stats.sample_failure_response, indent=2)
                if len(resp_str) > 500:
                    resp_str = resp_str[:500] + "... (truncated)"
                parts.append(f"- Sample Error Response:\n```json\n{resp_str}\n```")
            except Exception:
                parts.append(f"- Sample Error Response: {stats.sample_failure_response}")
        
        # Payload used
        if payload:
            parts.append(f"- Request Body: {json.dumps(payload.request_body)[:300]}")
        
        # OpenAPI spec excerpt
        path_spec = paths.get(step.endpoint, {})
        method_spec = path_spec.get(step.method.lower(), {})
        if method_spec:
            # Just include parameters and requestBody schema reference
            params = method_spec.get("parameters", [])
            if params:
                param_names = [p.get("name", "?") for p in params[:5]]
                parts.append(f"- Spec Parameters: {', '.join(param_names)}")
        
        parts.append("")
    
    return "\n".join(parts)


def _build_overall_summary(
    aggregated_result: AggregatedExecutionResult,
    consistently_failing: list[str],
    flaky_steps: list[str],
    batch_recaps: list[BatchRecap],
) -> str:
    """Build an overall summary from batch recaps."""
    total_steps = len(aggregated_result.step_stats)
    passing = len(aggregated_result.get_consistently_passing_steps())
    
    summary_parts = [
        f"Analyzed {total_steps} endpoints across {aggregated_result.num_runs} execution runs.",
        f"Results: {passing} consistently passing, {len(consistently_failing)} consistently failing, "
        f"{len(flaky_steps)} flaky.",
    ]
    
    # Collect common patterns across all batches
    all_patterns = []
    for recap in batch_recaps:
        all_patterns.extend(recap.common_patterns)
    
    if all_patterns:
        unique_patterns = list(set(all_patterns))[:5]
        summary_parts.append(f"Common patterns: {'; '.join(unique_patterns)}")
    
    return " ".join(summary_parts)


def _extract_priority_fixes(batch_recaps: list[BatchRecap]) -> list[str]:
    """Extract and prioritize fixes from batch recaps."""
    fixes: list[str] = []
    
    for recap in batch_recaps:
        for endpoint in recap.endpoint_recaps:
            if endpoint.confidence == "high":
                fixes.insert(0, f"[{endpoint.step_id}] {endpoint.suggested_fix}")
            else:
                fixes.append(f"[{endpoint.step_id}] {endpoint.suggested_fix}")
    
    return fixes[:10]  # Return top 10 priority fixes


def format_recap_for_refinement(recap: FullAnalysisRecap) -> str:
    """Format the recap as context for the refinement agent."""
    parts = [
        "# Pre-Analysis Summary (from multiple execution runs)",
        "",
        "## Overview",
        recap.overall_summary,
        "",
    ]
    
    if recap.flaky_step_notes:
        parts.extend([
            "## ⚠️ Flaky Tests Detected",
            recap.flaky_step_notes,
            "",
        ])
    
    if recap.priority_fixes:
        parts.extend([
            "## Priority Fixes",
            *[f"- {fix}" for fix in recap.priority_fixes[:10]],
            "",
        ])
    
    for i, batch in enumerate(recap.batch_recaps, 1):
        parts.append(f"## Batch {i} Analysis")
        parts.append(batch.batch_summary)
        parts.append("")
        
        for endpoint in batch.endpoint_recaps:
            parts.extend([
                f"### {endpoint.step_id} ({endpoint.method} {endpoint.endpoint})",
                f"- Issue: {endpoint.issue_summary}",
                f"- Cause: {endpoint.likely_cause}",
                f"- Fix: {endpoint.suggested_fix}",
                f"- Confidence: {endpoint.confidence}",
                "",
            ])
        
        if batch.common_patterns:
            parts.append("Common patterns in this batch:")
            parts.extend([f"- {p}" for p in batch.common_patterns])
            parts.append("")
    
    return "\n".join(parts)
