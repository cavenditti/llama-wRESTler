"""
Step classification phase: Classify steps as simple or complex for data generation.

Simple steps can use deterministic data generation (fast, no LLM).
Complex steps require LLM-based generation for proper handling.
"""

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llama_wrestler.models import APIPlan, APIStep, AuthRequirement, BodyFormat


logger = logging.getLogger(__name__)


class StepComplexity(str, Enum):
    """Classification of step complexity for data generation."""

    SIMPLE = "simple"  # Can use deterministic generation
    COMPLEX = "complex"  # Needs LLM-based generation


class ComplexityReason(str, Enum):
    """Reasons why a step is classified as complex."""

    HAS_DEPENDENCIES = "has_dependencies"  # Step depends on output from other steps
    IS_AUTH_PROVIDER = "is_auth_provider"  # Step provides auth (needs credentials)
    REQUIRES_AUTH = "requires_auth"  # Step needs auth token from another step
    MULTIPART_UPLOAD = "multipart_upload"  # File upload requires special handling
    COMPLEX_PAYLOAD = "complex_payload"  # Payload description suggests complexity
    REFERENCES_PREVIOUS = (
        "references_previous"  # Payload description references other steps
    )


class StepClassification(BaseModel):
    """Classification result for a single step."""

    step_id: str = Field(description="The step ID")
    complexity: StepComplexity = Field(
        description="Whether this step is simple or complex"
    )
    reasons: list[ComplexityReason] = Field(
        default_factory=list,
        description="Reasons for the complexity classification (empty for simple steps)",
    )


class ClassificationResult(BaseModel):
    """Result of classifying all steps in a test plan."""

    classifications: list[StepClassification] = Field(
        description="Classification for each step"
    )
    simple_count: int = Field(description="Number of simple steps")
    complex_count: int = Field(description="Number of complex steps")

    def get_simple_step_ids(self) -> set[str]:
        """Get the set of step IDs classified as simple."""
        return {
            c.step_id
            for c in self.classifications
            if c.complexity == StepComplexity.SIMPLE
        }

    def get_complex_step_ids(self) -> set[str]:
        """Get the set of step IDs classified as complex."""
        return {
            c.step_id
            for c in self.classifications
            if c.complexity == StepComplexity.COMPLEX
        }

    def is_simple(self, step_id: str) -> bool:
        """Check if a step is classified as simple."""
        return step_id in self.get_simple_step_ids()

    def is_complex(self, step_id: str) -> bool:
        """Check if a step is classified as complex."""
        return step_id in self.get_complex_step_ids()


def _check_payload_references_steps(
    payload_description: str | None, all_step_ids: set[str]
) -> bool:
    """
    Check if a payload description references other steps.

    Looks for patterns like:
    - "use ID from step X"
    - "reference {{step_id}}"
    - "from create_user response"
    """
    if not payload_description:
        return False

    desc_lower = payload_description.lower()

    # Check for explicit step references
    for step_id in all_step_ids:
        if step_id.lower() in desc_lower:
            return True

    # Check for common patterns indicating step references
    reference_patterns = [
        r"from\s+(the\s+)?(previous|other|another)\s+step",
        r"use\s+(the\s+)?id\s+from",
        r"reference\s+(the\s+)?response",
        r"\{\{[^}]+\}\}",  # Placeholder syntax
        r"created\s+(by|in)\s+(the\s+)?previous",
        r"returned\s+(by|from)",
    ]

    for pattern in reference_patterns:
        if re.search(pattern, desc_lower):
            return True

    return False


def _check_complex_payload(payload_description: str | None) -> bool:
    """
    Check if a payload description suggests complexity that needs LLM.

    Patterns that suggest complexity:
    - Conditional logic
    - Complex relationships
    - Dynamic values
    """
    if not payload_description:
        return False

    desc_lower = payload_description.lower()

    complexity_patterns = [
        r"depending\s+on",
        r"based\s+on",
        r"conditional",
        r"if\s+.*\s+then",
        r"must\s+match",
        r"should\s+be\s+unique",
        r"must\s+be\s+valid",
        r"format.*specific",
    ]

    for pattern in complexity_patterns:
        if re.search(pattern, desc_lower):
            return True

    return False


def classify_step(
    step: APIStep,
    all_step_ids: set[str],
    has_credentials: bool = False,
) -> StepClassification:
    """
    Classify a single step as simple or complex.

    Args:
        step: The API step to classify
        all_step_ids: Set of all step IDs in the plan (for reference detection)
        has_credentials: Whether credentials are available

    Returns:
        StepClassification with complexity and reasons
    """
    reasons: list[ComplexityReason] = []

    # Check for dependencies
    if step.depends_on:
        reasons.append(ComplexityReason.HAS_DEPENDENCIES)

    # Check for auth provider status
    if step.auth_requirement == AuthRequirement.AUTH_PROVIDER:
        # Auth provider steps need credentials injection
        if has_credentials:
            reasons.append(ComplexityReason.IS_AUTH_PROVIDER)

    # NOTE: Steps that require auth (auth_requirement=REQUIRED) are NOT marked complex
    # because the deterministic generator can add the auth header placeholder.
    # The auth token placeholder is a simple string substitution.

    # Check for multipart (file upload)
    if step.body_format == BodyFormat.MULTIPART:
        reasons.append(ComplexityReason.MULTIPART_UPLOAD)

    # Check if payload description references other steps
    if _check_payload_references_steps(step.payload_description, all_step_ids):
        reasons.append(ComplexityReason.REFERENCES_PREVIOUS)

    # Check for complex payload requirements
    if _check_complex_payload(step.payload_description):
        reasons.append(ComplexityReason.COMPLEX_PAYLOAD)

    # Determine complexity based on reasons
    complexity = StepComplexity.COMPLEX if reasons else StepComplexity.SIMPLE

    return StepClassification(
        step_id=step.id,
        complexity=complexity,
        reasons=reasons,
    )


def classify_steps(
    test_plan: APIPlan,
    has_credentials: bool = False,
) -> ClassificationResult:
    """
    Classify all steps in a test plan.

    Args:
        test_plan: The test plan to classify
        has_credentials: Whether credentials are available for auth steps

    Returns:
        ClassificationResult with all step classifications
    """
    all_step_ids = {step.id for step in test_plan.steps}

    classifications = [
        classify_step(step, all_step_ids, has_credentials) for step in test_plan.steps
    ]

    simple_count = sum(
        1 for c in classifications if c.complexity == StepComplexity.SIMPLE
    )
    complex_count = len(classifications) - simple_count

    result = ClassificationResult(
        classifications=classifications,
        simple_count=simple_count,
        complex_count=complex_count,
    )

    logger.info(
        "Step classification: %d simple, %d complex (total: %d)",
        simple_count,
        complex_count,
        len(classifications),
    )

    # Log details for complex steps
    for classification in classifications:
        if classification.complexity == StepComplexity.COMPLEX:
            reason_str = ", ".join(r.value for r in classification.reasons)
            logger.debug(
                "Complex step '%s': %s",
                classification.step_id,
                reason_str,
            )

    return result


def run_classification_phase(
    test_plan: APIPlan,
    credentials: Any | None = None,
) -> ClassificationResult:
    """
    Run the step classification phase.

    This phase analyzes each step in the test plan and classifies it as:
    - SIMPLE: Can use fast deterministic data generation
    - COMPLEX: Needs LLM-based generation for proper handling

    Args:
        test_plan: The test plan from the preliminary phase
        credentials: Optional credentials (affects auth step classification)

    Returns:
        ClassificationResult with classifications for all steps
    """
    has_credentials = credentials is not None
    return classify_steps(test_plan, has_credentials)
