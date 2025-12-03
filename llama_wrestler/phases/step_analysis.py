"""
Step analysis phase: Analyze steps for dependency fields that need placeholders.

This module identifies which fields in each step's payload need to reference
data from previous steps. The strong LLM is used once to analyze all steps
and identify placeholder requirements, then deterministic generation fills
in the rest.

The goal is to minimize LLM usage while still handling inter-step dependencies
correctly.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from llama_wrestler.models import APIPlan, APIStep, AuthRequirement, BodyFormat


logger = logging.getLogger(__name__)


class FieldPlaceholder(BaseModel):
    """A placeholder reference for a field that depends on a previous step."""
    
    field_path: str = Field(
        description="Path to the field in the payload (e.g., 'user_id', 'data.parent_id')"
    )
    source_step_id: str = Field(
        description="ID of the step to get the value from"
    )
    source_field_path: str = Field(
        description="Path to the field in the source step's response (e.g., 'id', 'data.id')"
    )
    
    def to_placeholder(self) -> str:
        """Convert to placeholder syntax."""
        return f"{{{{{self.source_step_id}.{self.source_field_path}}}}}"


class StepAnalysis(BaseModel):
    """Analysis result for a single step."""
    
    step_id: str = Field(description="The step ID")
    has_dependencies: bool = Field(
        default=False,
        description="Whether this step depends on other steps"
    )
    dependency_step_ids: list[str] = Field(
        default_factory=list,
        description="IDs of steps this step depends on"
    )
    field_placeholders: list[FieldPlaceholder] = Field(
        default_factory=list,
        description="Fields that need placeholder values from previous steps"
    )
    needs_credentials: bool = Field(
        default=False,
        description="Whether this step needs credential injection"
    )
    is_multipart: bool = Field(
        default=False,
        description="Whether this step uses multipart/file upload"
    )


class AnalysisResult(BaseModel):
    """Result of analyzing all steps in a test plan."""
    
    analyses: list[StepAnalysis] = Field(
        description="Analysis for each step"
    )
    steps_with_dependencies: int = Field(
        default=0,
        description="Number of steps that have dependencies"
    )
    total_placeholders: int = Field(
        default=0,
        description="Total number of placeholder fields identified"
    )
    
    def get_analysis(self, step_id: str) -> StepAnalysis | None:
        """Get analysis for a specific step."""
        for analysis in self.analyses:
            if analysis.step_id == step_id:
                return analysis
        return None
    
    def get_placeholders(self, step_id: str) -> list[FieldPlaceholder]:
        """Get placeholder fields for a specific step."""
        analysis = self.get_analysis(step_id)
        return analysis.field_placeholders if analysis else []


def _extract_path_params(endpoint: str) -> list[str]:
    """Extract path parameter names from an endpoint template."""
    return re.findall(r"\{(\w+)\}", endpoint)


def _find_auth_provider_step(test_plan: APIPlan) -> str | None:
    """Find the first auth provider step in the plan."""
    for step in test_plan.steps:
        if step.auth_requirement == AuthRequirement.AUTH_PROVIDER:
            return step.id
    return None


def analyze_step_dependencies(
    step: APIStep,
    test_plan: APIPlan,
    has_credentials: bool = False,
) -> StepAnalysis:
    """
    Analyze a step's dependencies and identify placeholder fields.
    
    This performs static analysis based on:
    - Step dependencies (depends_on field)
    - Path parameters that might reference previous steps
    - Auth requirements
    
    Args:
        step: The API step to analyze
        test_plan: The full test plan for context
        has_credentials: Whether credentials are available
        
    Returns:
        StepAnalysis with dependency information
    """
    analysis = StepAnalysis(step_id=step.id)
    
    # Track dependencies
    if step.depends_on:
        analysis.has_dependencies = True
        analysis.dependency_step_ids = list(step.depends_on)
    
    # Check for credential needs
    if step.auth_requirement == AuthRequirement.AUTH_PROVIDER and has_credentials:
        analysis.needs_credentials = True
    
    # Check for multipart
    if step.body_format == BodyFormat.MULTIPART:
        analysis.is_multipart = True
    
    # Identify path parameters that likely need placeholders
    path_params = _extract_path_params(step.endpoint)
    if path_params and step.depends_on:
        # If we have path params and dependencies, those params likely
        # reference data from dependency steps
        for param in path_params:
            # Common patterns: {id}, {user_id}, {item_id}
            # Try to match to a dependency step
            for dep_id in step.depends_on:
                # Heuristic: if param contains common ID patterns
                if param in ("id", "Id", "ID") or param.endswith("_id") or param.endswith("Id"):
                    analysis.field_placeholders.append(
                        FieldPlaceholder(
                            field_path=f"path_params.{param}",
                            source_step_id=dep_id,
                            source_field_path="id",  # Common default
                        )
                    )
                    break  # Only add once per param
    
    # Add auth header placeholder for protected endpoints
    if step.auth_requirement == AuthRequirement.REQUIRED:
        auth_step_id = _find_auth_provider_step(test_plan)
        if auth_step_id:
            analysis.field_placeholders.append(
                FieldPlaceholder(
                    field_path="headers.Authorization",
                    source_step_id=auth_step_id,
                    source_field_path="access_token",
                )
            )
    
    return analysis


def run_analysis_phase(
    test_plan: APIPlan,
    credentials: Any | None = None,
) -> AnalysisResult:
    """
    Run the step analysis phase.
    
    This phase performs static analysis of steps to identify:
    - Dependencies between steps
    - Fields that need placeholder values from previous steps
    - Special handling requirements (credentials, multipart)
    
    This is a lightweight phase that doesn't use the LLM.
    The results inform the data generation phase about which
    fields need special handling.
    
    Args:
        test_plan: The test plan from the preliminary phase
        credentials: Optional credentials
        
    Returns:
        AnalysisResult with analysis for all steps
    """
    has_credentials = credentials is not None
    
    analyses = [
        analyze_step_dependencies(step, test_plan, has_credentials)
        for step in test_plan.steps
    ]
    
    steps_with_deps = sum(1 for a in analyses if a.has_dependencies)
    total_placeholders = sum(len(a.field_placeholders) for a in analyses)
    
    result = AnalysisResult(
        analyses=analyses,
        steps_with_dependencies=steps_with_deps,
        total_placeholders=total_placeholders,
    )
    
    logger.info(
        "Step analysis: %d/%d steps have dependencies, %d total placeholder fields",
        steps_with_deps,
        len(analyses),
        total_placeholders,
    )
    
    return result
