"""
Tests for step classification module.
"""

from llama_wrestler.phases.step_classification import (
    classify_step,
    classify_steps,
    run_classification_phase,
    StepComplexity,
    ComplexityReason,
    ClassificationResult,
    StepClassification,
)
from llama_wrestler.models import (
    APIPlan,
    APIStep,
    AuthRequirement,
    BodyFormat,
    APICredentials,
)


class TestClassifyStep:
    """Tests for classify_step function."""

    def test_simple_get_no_deps(self):
        """Test that a simple GET with no dependencies is classified as simple."""
        step = APIStep(
            id="get_users",
            description="Get all users",
            endpoint="/users",
            method="GET",
            expected_status=200,
            auth_requirement=AuthRequirement.NONE,
        )

        result = classify_step(step, {"get_users"})

        assert result.complexity == StepComplexity.SIMPLE
        assert result.reasons == []

    def test_simple_post_no_deps(self):
        """Test that a simple POST with no dependencies is classified as simple."""
        step = APIStep(
            id="create_user",
            description="Create a user",
            endpoint="/users",
            method="POST",
            expected_status=201,
            auth_requirement=AuthRequirement.NONE,
        )

        result = classify_step(step, {"create_user"})

        assert result.complexity == StepComplexity.SIMPLE
        assert result.reasons == []

    def test_complex_with_dependencies(self):
        """Test that a step with dependencies is classified as complex."""
        step = APIStep(
            id="get_user",
            description="Get specific user",
            endpoint="/users/{id}",
            method="GET",
            expected_status=200,
            depends_on=["create_user"],
            auth_requirement=AuthRequirement.NONE,
        )

        result = classify_step(step, {"create_user", "get_user"})

        assert result.complexity == StepComplexity.COMPLEX
        assert ComplexityReason.HAS_DEPENDENCIES in result.reasons

    def test_complex_auth_provider_with_creds(self):
        """Test that auth provider with credentials is complex."""
        step = APIStep(
            id="login",
            description="Login",
            endpoint="/auth/login",
            method="POST",
            expected_status=200,
            auth_requirement=AuthRequirement.AUTH_PROVIDER,
        )

        result = classify_step(step, {"login"}, has_credentials=True)

        assert result.complexity == StepComplexity.COMPLEX
        assert ComplexityReason.IS_AUTH_PROVIDER in result.reasons

    def test_simple_auth_provider_no_creds(self):
        """Test that auth provider without credentials is simple (uses generated creds)."""
        step = APIStep(
            id="login",
            description="Login",
            endpoint="/auth/login",
            method="POST",
            expected_status=200,
            auth_requirement=AuthRequirement.AUTH_PROVIDER,
        )

        result = classify_step(step, {"login"}, has_credentials=False)

        # Without credentials, auth_provider doesn't need special handling
        assert result.complexity == StepComplexity.SIMPLE

    def test_simple_requires_auth(self):
        """Test that requiring auth alone is simple (deterministic can add auth header)."""
        step = APIStep(
            id="get_profile",
            description="Get user profile",
            endpoint="/profile",
            method="GET",
            expected_status=200,
            auth_requirement=AuthRequirement.REQUIRED,
        )

        result = classify_step(step, {"get_profile"})

        # Auth requirement alone doesn't make a step complex
        # The deterministic generator can add the auth header placeholder
        assert result.complexity == StepComplexity.SIMPLE
        assert result.reasons == []

    def test_complex_multipart(self):
        """Test that multipart uploads are complex."""
        step = APIStep(
            id="upload_file",
            description="Upload a file",
            endpoint="/files",
            method="POST",
            expected_status=201,
            body_format=BodyFormat.MULTIPART,
        )

        result = classify_step(step, {"upload_file"})

        assert result.complexity == StepComplexity.COMPLEX
        assert ComplexityReason.MULTIPART_UPLOAD in result.reasons

    def test_complex_payload_references_previous(self):
        """Test that payload referencing previous steps is complex."""
        step = APIStep(
            id="update_user",
            description="Update the user",
            endpoint="/users/{id}",
            method="PUT",
            expected_status=200,
            payload_description="Use the ID from create_user response",
        )

        result = classify_step(step, {"create_user", "update_user"})

        assert result.complexity == StepComplexity.COMPLEX
        assert ComplexityReason.REFERENCES_PREVIOUS in result.reasons

    def test_complex_payload_with_placeholder(self):
        """Test that payload with placeholder syntax is complex."""
        step = APIStep(
            id="delete_user",
            description="Delete user",
            endpoint="/users/{id}",
            method="DELETE",
            expected_status=204,
            payload_description="Use {{create_user.id}} as the user ID",
        )

        result = classify_step(step, {"create_user", "delete_user"})

        assert result.complexity == StepComplexity.COMPLEX
        assert ComplexityReason.REFERENCES_PREVIOUS in result.reasons

    def test_multiple_complexity_reasons(self):
        """Test that multiple reasons can be present."""
        step = APIStep(
            id="upload_avatar",
            description="Upload avatar",
            endpoint="/users/{id}/avatar",
            method="POST",
            expected_status=200,
            depends_on=["create_user"],
            body_format=BodyFormat.MULTIPART,
        )

        result = classify_step(step, {"create_user", "upload_avatar"})

        assert result.complexity == StepComplexity.COMPLEX
        assert ComplexityReason.HAS_DEPENDENCIES in result.reasons
        assert ComplexityReason.MULTIPART_UPLOAD in result.reasons


class TestClassifySteps:
    """Tests for classify_steps function."""

    def test_all_simple(self):
        """Test classification of all simple steps."""
        plan = APIPlan(
            summary="Test plan",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="a",
                    description="A",
                    endpoint="/a",
                    method="GET",
                    expected_status=200,
                ),
                APIStep(
                    id="b",
                    description="B",
                    endpoint="/b",
                    method="POST",
                    expected_status=201,
                ),
            ],
        )

        result = classify_steps(plan)

        assert result.simple_count == 2
        assert result.complex_count == 0

    def test_mixed_simple_complex(self):
        """Test classification of mixed simple and complex steps."""
        plan = APIPlan(
            summary="Test plan",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="a",
                    description="A",
                    endpoint="/a",
                    method="GET",
                    expected_status=200,
                ),
                APIStep(
                    id="b",
                    description="B",
                    endpoint="/b",
                    method="POST",
                    expected_status=201,
                    depends_on=["a"],
                ),
            ],
        )

        result = classify_steps(plan)

        assert result.simple_count == 1
        assert result.complex_count == 1

    def test_get_simple_step_ids(self):
        """Test getting set of simple step IDs."""
        plan = APIPlan(
            summary="Test plan",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="simple1",
                    description="S1",
                    endpoint="/s1",
                    method="GET",
                    expected_status=200,
                ),
                APIStep(
                    id="complex1",
                    description="C1",
                    endpoint="/c1",
                    method="POST",
                    expected_status=200,
                    depends_on=["simple1"],
                ),
                APIStep(
                    id="simple2",
                    description="S2",
                    endpoint="/s2",
                    method="GET",
                    expected_status=200,
                ),
            ],
        )

        result = classify_steps(plan)
        simple_ids = result.get_simple_step_ids()

        assert simple_ids == {"simple1", "simple2"}

    def test_get_complex_step_ids(self):
        """Test getting set of complex step IDs."""
        plan = APIPlan(
            summary="Test plan",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="simple1",
                    description="S1",
                    endpoint="/s1",
                    method="GET",
                    expected_status=200,
                ),
                APIStep(
                    id="complex1",
                    description="C1",
                    endpoint="/c1",
                    method="POST",
                    expected_status=200,
                    depends_on=["simple1"],
                ),
            ],
        )

        result = classify_steps(plan)
        complex_ids = result.get_complex_step_ids()

        assert complex_ids == {"complex1"}


class TestClassificationResult:
    """Tests for ClassificationResult methods."""

    def test_is_simple(self):
        """Test is_simple method."""
        result = ClassificationResult(
            classifications=[
                StepClassification(
                    step_id="a", complexity=StepComplexity.SIMPLE, reasons=[]
                ),
                StepClassification(
                    step_id="b",
                    complexity=StepComplexity.COMPLEX,
                    reasons=[ComplexityReason.HAS_DEPENDENCIES],
                ),
            ],
            simple_count=1,
            complex_count=1,
        )

        assert result.is_simple("a") is True
        assert result.is_simple("b") is False
        assert result.is_simple("c") is False

    def test_is_complex(self):
        """Test is_complex method."""
        result = ClassificationResult(
            classifications=[
                StepClassification(
                    step_id="a", complexity=StepComplexity.SIMPLE, reasons=[]
                ),
                StepClassification(
                    step_id="b",
                    complexity=StepComplexity.COMPLEX,
                    reasons=[ComplexityReason.HAS_DEPENDENCIES],
                ),
            ],
            simple_count=1,
            complex_count=1,
        )

        assert result.is_complex("a") is False
        assert result.is_complex("b") is True


class TestRunClassificationPhase:
    """Tests for run_classification_phase function."""

    def test_with_credentials(self):
        """Test that credentials affect classification."""
        plan = APIPlan(
            summary="Test",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="login",
                    description="Login",
                    endpoint="/login",
                    method="POST",
                    expected_status=200,
                    auth_requirement=AuthRequirement.AUTH_PROVIDER,
                ),
            ],
        )

        creds = APICredentials(username="test", password="pass")
        result = run_classification_phase(plan, credentials=creds)

        assert result.complex_count == 1
        assert result.is_complex("login")

    def test_without_credentials(self):
        """Test classification without credentials."""
        plan = APIPlan(
            summary="Test",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="login",
                    description="Login",
                    endpoint="/login",
                    method="POST",
                    expected_status=200,
                    auth_requirement=AuthRequirement.AUTH_PROVIDER,
                ),
            ],
        )

        result = run_classification_phase(plan, credentials=None)

        # Without credentials, auth_provider doesn't trigger complexity
        assert result.simple_count == 1


class TestComplexPayloadDetection:
    """Tests for complex payload description detection."""

    def test_conditional_payload(self):
        """Test that conditional payloads are detected."""
        step = APIStep(
            id="test",
            description="Test",
            endpoint="/test",
            method="POST",
            expected_status=200,
            payload_description="Include field X depending on user type",
        )

        result = classify_step(step, {"test"})

        assert ComplexityReason.COMPLEX_PAYLOAD in result.reasons

    def test_based_on_payload(self):
        """Test that 'based on' payloads are detected."""
        step = APIStep(
            id="test",
            description="Test",
            endpoint="/test",
            method="POST",
            expected_status=200,
            payload_description="Set value based on previous response",
        )

        result = classify_step(step, {"test"})

        assert ComplexityReason.COMPLEX_PAYLOAD in result.reasons
