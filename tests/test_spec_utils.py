"""
Tests for spec_utils module.
"""

from pathlib import Path
import tempfile

from llama_wrestler.spec_utils import (
    normalize_spec,
    compute_spec_hash,
    save_spec_hash,
    load_spec_hash,
    find_cached_test_plan,
    topological_sort_steps,
    sort_api_plan,
    extract_security_requirements,
    get_security_definitions,
    validate_auth_requirements,
    suggest_auth_provider_steps,
)
from llama_wrestler.models import APIPlan, APIStep, AuthRequirement


class TestNormalizeSpec:
    """Tests for normalize_spec function."""

    def test_normalize_sorts_keys(self):
        """Test that dictionary keys are sorted."""
        spec = {"z": 1, "a": 2, "m": 3}
        result = normalize_spec(spec)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_normalize_removes_description(self):
        """Test that description fields are removed."""
        spec = {
            "info": {"title": "Test API", "description": "This should be removed"},
            "paths": {},
        }
        result = normalize_spec(spec)
        assert "description" not in result.get("info", {})
        assert result["info"]["title"] == "Test API"

    def test_normalize_removes_examples(self):
        """Test that example fields are removed."""
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "responses": {
                            "200": {"example": {"id": 1}, "examples": [{"id": 2}]}
                        }
                    }
                }
            }
        }
        result = normalize_spec(spec)
        response = result["paths"]["/users"]["get"]["responses"]["200"]
        assert "example" not in response
        assert "examples" not in response

    def test_normalize_handles_nested_structures(self):
        """Test that nested structures are properly normalized."""
        spec = {
            "paths": {
                "/users": {"get": {"parameters": [{"name": "id", "in": "query"}]}}
            }
        }
        result = normalize_spec(spec)
        assert result["paths"]["/users"]["get"]["parameters"][0]["name"] == "id"

    def test_normalize_strips_whitespace_from_strings(self):
        """Test that string values are stripped."""
        spec = {"title": "  Test API  "}
        result = normalize_spec(spec)
        assert result["title"] == "Test API"


class TestComputeSpecHash:
    """Tests for compute_spec_hash function."""

    def test_same_spec_produces_same_hash(self):
        """Test that identical specs produce identical hashes."""
        spec = {"info": {"title": "Test"}, "paths": {}}
        hash1 = compute_spec_hash(spec)
        hash2 = compute_spec_hash(spec)
        assert hash1 == hash2

    def test_different_key_order_produces_same_hash(self):
        """Test that key order doesn't affect hash."""
        spec1 = {"a": 1, "b": 2}
        spec2 = {"b": 2, "a": 1}
        assert compute_spec_hash(spec1) == compute_spec_hash(spec2)

    def test_description_changes_dont_affect_hash(self):
        """Test that changing description doesn't change hash."""
        spec1 = {"info": {"title": "Test", "description": "Version 1"}}
        spec2 = {"info": {"title": "Test", "description": "Version 2"}}
        assert compute_spec_hash(spec1) == compute_spec_hash(spec2)

    def test_path_changes_affect_hash(self):
        """Test that changing paths does change hash."""
        spec1 = {"paths": {"/users": {}}}
        spec2 = {"paths": {"/posts": {}}}
        assert compute_spec_hash(spec1) != compute_spec_hash(spec2)

    def test_hash_is_sha256_hex(self):
        """Test that hash is a valid SHA-256 hex string."""
        spec = {"test": "data"}
        hash_value = compute_spec_hash(spec)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestSaveLoadSpecHash:
    """Tests for save_spec_hash and load_spec_hash functions."""

    def test_save_and_load_hash(self):
        """Test saving and loading a hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            test_hash = "abc123def456"

            save_spec_hash(run_dir, test_hash)
            loaded = load_spec_hash(run_dir)

            assert loaded == test_hash

    def test_load_nonexistent_returns_none(self):
        """Test that loading from nonexistent directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "nonexistent"
            assert load_spec_hash(run_dir) is None


class TestFindCachedTestPlan:
    """Tests for find_cached_test_plan function."""

    def test_find_cached_plan_returns_matching(self):
        """Test finding a cached plan with matching hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            run_dir = output_dir / "001_test_20240101_120000"
            run_dir.mkdir()

            # Create hash file
            test_hash = "abc123"
            save_spec_hash(run_dir, test_hash)

            # Create test plan
            plan = APIPlan(summary="Test plan", base_url="http://localhost", steps=[])
            with open(run_dir / "test_plan.json", "w") as f:
                f.write(plan.model_dump_json())

            # Find cached plan
            cached = find_cached_test_plan(output_dir, test_hash)
            assert cached is not None
            assert cached.summary == "Test plan"

    def test_find_cached_plan_returns_none_for_no_match(self):
        """Test that no match returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            run_dir = output_dir / "001_test_20240101_120000"
            run_dir.mkdir()

            save_spec_hash(run_dir, "different_hash")

            cached = find_cached_test_plan(output_dir, "abc123")
            assert cached is None

    def test_find_cached_plan_returns_latest(self):
        """Test that the latest matching run is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            test_hash = "abc123"

            # Create older run
            run_dir1 = output_dir / "001_test_20240101_120000"
            run_dir1.mkdir()
            save_spec_hash(run_dir1, test_hash)
            plan1 = APIPlan(summary="Old plan", base_url="http://localhost", steps=[])
            with open(run_dir1 / "test_plan.json", "w") as f:
                f.write(plan1.model_dump_json())

            # Create newer run
            run_dir2 = output_dir / "002_test_20240102_120000"
            run_dir2.mkdir()
            save_spec_hash(run_dir2, test_hash)
            plan2 = APIPlan(summary="New plan", base_url="http://localhost", steps=[])
            with open(run_dir2 / "test_plan.json", "w") as f:
                f.write(plan2.model_dump_json())

            cached = find_cached_test_plan(output_dir, test_hash)
            assert cached.summary == "New plan"


class TestTopologicalSort:
    """Tests for topological_sort_steps function."""

    def test_sort_no_dependencies(self):
        """Test sorting steps with no dependencies."""
        steps = [
            APIStep(
                id="c_step",
                description="C",
                endpoint="/c",
                method="GET",
                expected_status=200,
            ),
            APIStep(
                id="a_step",
                description="A",
                endpoint="/a",
                method="GET",
                expected_status=200,
            ),
            APIStep(
                id="b_step",
                description="B",
                endpoint="/b",
                method="GET",
                expected_status=200,
            ),
        ]

        sorted_steps = topological_sort_steps(steps)
        ids = [s.id for s in sorted_steps]

        # Should be lexicographically sorted
        assert ids == ["a_step", "b_step", "c_step"]

    def test_sort_with_dependencies(self):
        """Test sorting with dependencies."""
        steps = [
            APIStep(
                id="step_3",
                description="3",
                endpoint="/3",
                method="GET",
                expected_status=200,
                depends_on=["step_2"],
            ),
            APIStep(
                id="step_1",
                description="1",
                endpoint="/1",
                method="GET",
                expected_status=200,
            ),
            APIStep(
                id="step_2",
                description="2",
                endpoint="/2",
                method="GET",
                expected_status=200,
                depends_on=["step_1"],
            ),
        ]

        sorted_steps = topological_sort_steps(steps)
        ids = [s.id for s in sorted_steps]

        # step_1 must come before step_2, step_2 before step_3
        assert ids.index("step_1") < ids.index("step_2")
        assert ids.index("step_2") < ids.index("step_3")

    def test_sort_lexicographical_tiebreaker(self):
        """Test that lexicographical order is used as tiebreaker."""
        steps = [
            APIStep(
                id="z_step",
                description="Z",
                endpoint="/z",
                method="GET",
                expected_status=200,
            ),
            APIStep(
                id="a_step",
                description="A",
                endpoint="/a",
                method="GET",
                expected_status=200,
            ),
            APIStep(
                id="m_step",
                description="M",
                endpoint="/m",
                method="GET",
                expected_status=200,
                depends_on=["a_step", "z_step"],
            ),
        ]

        sorted_steps = topological_sort_steps(steps)
        ids = [s.id for s in sorted_steps]

        # a_step and z_step have no deps, so sorted lexicographically
        assert ids[0] == "a_step"
        assert ids[1] == "z_step"
        assert ids[2] == "m_step"

    def test_sort_handles_missing_dependency(self):
        """Test that missing dependencies are handled gracefully."""
        steps = [
            APIStep(
                id="step_1",
                description="1",
                endpoint="/1",
                method="GET",
                expected_status=200,
                depends_on=["nonexistent"],
            ),
        ]

        sorted_steps = topological_sort_steps(steps)
        assert len(sorted_steps) == 1
        assert sorted_steps[0].id == "step_1"


class TestSortApiPlan:
    """Tests for sort_api_plan function."""

    def test_sort_preserves_metadata(self):
        """Test that sorting preserves plan metadata."""
        plan = APIPlan(
            summary="Test Summary",
            base_url="http://test.com",
            steps=[
                APIStep(
                    id="b",
                    description="B",
                    endpoint="/b",
                    method="GET",
                    expected_status=200,
                ),
                APIStep(
                    id="a",
                    description="A",
                    endpoint="/a",
                    method="GET",
                    expected_status=200,
                ),
            ],
        )

        sorted_plan = sort_api_plan(plan)

        assert sorted_plan.summary == "Test Summary"
        assert sorted_plan.base_url == "http://test.com"
        assert sorted_plan.steps[0].id == "a"


class TestExtractSecurityRequirements:
    """Tests for extract_security_requirements function."""

    def test_extract_swagger2_security(self):
        """Test extracting security from Swagger 2.0 spec."""
        spec = {
            "swagger": "2.0",
            "paths": {"/users": {"get": {"security": [{"api_key": []}]}}},
        }

        reqs = extract_security_requirements(spec)
        assert "GET /users" in reqs
        assert "api_key" in reqs["GET /users"]

    def test_extract_openapi3_security(self):
        """Test extracting security from OpenAPI 3.x spec."""
        spec = {
            "openapi": "3.0.0",
            "paths": {"/users": {"get": {"security": [{"bearerAuth": []}]}}},
        }

        reqs = extract_security_requirements(spec)
        assert "GET /users" in reqs
        assert "bearerAuth" in reqs["GET /users"]

    def test_extract_global_security(self):
        """Test that global security is applied to endpoints."""
        spec = {
            "openapi": "3.0.0",
            "security": [{"bearerAuth": []}],
            "paths": {"/users": {"get": {}}},
        }

        reqs = extract_security_requirements(spec)
        assert "bearerAuth" in reqs["GET /users"]

    def test_empty_security_overrides_global(self):
        """Test that empty security array means no auth required."""
        spec = {
            "openapi": "3.0.0",
            "security": [{"bearerAuth": []}],
            "paths": {"/public": {"get": {"security": []}}},
        }

        reqs = extract_security_requirements(spec)
        assert reqs["GET /public"] == set()


class TestValidateAuthRequirements:
    """Tests for validate_auth_requirements function."""

    def test_warns_when_plan_says_none_but_spec_requires(self):
        """Test warning when plan has no auth but spec requires it."""
        plan = APIPlan(
            summary="Test",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="get_users",
                    description="Get users",
                    endpoint="/users",
                    method="GET",
                    expected_status=200,
                    auth_requirement=AuthRequirement.NONE,
                )
            ],
        )

        spec = {
            "openapi": "3.0.0",
            "paths": {"/users": {"get": {"security": [{"bearerAuth": []}]}}},
        }

        warnings = validate_auth_requirements(plan, spec)
        assert len(warnings) == 1
        assert "auth_requirement=NONE" in warnings[0]

    def test_no_warning_when_both_require_auth(self):
        """Test no warning when both plan and spec require auth."""
        plan = APIPlan(
            summary="Test",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="get_users",
                    description="Get users",
                    endpoint="/users",
                    method="GET",
                    expected_status=200,
                    auth_requirement=AuthRequirement.REQUIRED,
                )
            ],
        )

        spec = {
            "openapi": "3.0.0",
            "paths": {"/users": {"get": {"security": [{"bearerAuth": []}]}}},
        }

        warnings = validate_auth_requirements(plan, spec)
        assert len(warnings) == 0

    def test_handles_path_parameters(self):
        """Test that path parameters are matched correctly."""
        plan = APIPlan(
            summary="Test",
            base_url="http://localhost",
            steps=[
                APIStep(
                    id="get_user",
                    description="Get user",
                    endpoint="/users/123",
                    method="GET",
                    expected_status=200,
                    auth_requirement=AuthRequirement.NONE,
                )
            ],
        )

        spec = {
            "openapi": "3.0.0",
            "paths": {"/users/{id}": {"get": {"security": [{"bearerAuth": []}]}}},
        }

        warnings = validate_auth_requirements(plan, spec)
        assert len(warnings) == 1


class TestSuggestAuthProviderSteps:
    """Tests for suggest_auth_provider_steps function."""

    def test_identifies_login_endpoints(self):
        """Test that login endpoints are identified."""
        spec = {
            "paths": {
                "/auth/login": {"post": {}},
                "/users": {"get": {}},
            }
        }

        auth_endpoints = suggest_auth_provider_steps(spec)
        assert "POST /auth/login" in auth_endpoints

    def test_identifies_token_endpoints(self):
        """Test that OAuth token endpoints are identified."""
        spec = {
            "paths": {
                "/oauth/token": {"post": {}},
            }
        }

        auth_endpoints = suggest_auth_provider_steps(spec)
        assert "POST /oauth/token" in auth_endpoints

    def test_ignores_non_auth_endpoints(self):
        """Test that non-auth endpoints are not included."""
        spec = {
            "paths": {
                "/users": {"get": {}, "post": {}},
            }
        }

        auth_endpoints = suggest_auth_provider_steps(spec)
        assert len(auth_endpoints) == 0


class TestGetSecurityDefinitions:
    """Tests for get_security_definitions function."""

    def test_swagger2_security_definitions(self):
        """Test getting security definitions from Swagger 2.0."""
        spec = {
            "swagger": "2.0",
            "securityDefinitions": {
                "api_key": {"type": "apiKey", "name": "api_key", "in": "header"}
            },
        }

        defs = get_security_definitions(spec)
        assert "api_key" in defs

    def test_openapi3_security_schemes(self):
        """Test getting security schemes from OpenAPI 3.x."""
        spec = {
            "openapi": "3.0.0",
            "components": {
                "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}}
            },
        }

        defs = get_security_definitions(spec)
        assert "bearerAuth" in defs
