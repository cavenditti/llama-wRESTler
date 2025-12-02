import asyncio

import httpx

from llama_wrestler.models import AuthRequirement, BodyFormat, APIPlan, APIStep
from llama_wrestler.phases.data_generation import GeneratedTestData, MockedPayload
from llama_wrestler.phases.test_execution import (
    ExecutionContext,
    build_url,
    execute_step,
    prepare_request_body,
    resolve_placeholders,
    run_test_execution_phase,
)


def test_build_url_encodes_params():
    url = build_url(
        "https://api.test/",
        "/files/{name}",
        {"name": "report 1"},
        {"tag": "a/b"},
    )

    assert url == "https://api.test/files/report%201?tag=a%2Fb"


def test_prepare_request_body_preserves_empty_json():
    body = prepare_request_body(BodyFormat.JSON, {})

    assert body.json == {}
    assert body.headers["Content-Type"] == "application/json"


def test_resolve_placeholders_handles_hyphenated_ids():
    resolved = resolve_placeholders(
        "Bearer {{auth-step.token}} to {{auth-step.nested.id}}",
        {"auth-step": {"token": "abc123", "nested": {"id": 7}}},
    )
    assert resolved == "Bearer abc123 to 7"

    unresolved = resolve_placeholders(
        "Still missing: {{missing.value}}",
        {"auth-step": {"token": "abc123"}},
    )
    assert unresolved == "Still missing: {{missing.value}}"


def test_execute_step_applies_auth_header():
    async def _run():
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            ctx = ExecutionContext(http_client=client, base_url="https://api.test")
            ctx.auth_tokens["login"] = "t-123"
            step = APIStep(
                id="protected",
                description="protected endpoint",
                endpoint="/ping",
                method="GET",
                expected_status=200,
                depends_on=["login"],
                body_format=BodyFormat.NONE,
                auth_requirement=AuthRequirement.REQUIRED,
            )
            payload = MockedPayload(step_id="protected", request_body=None)

            result = await execute_step(step, payload, ctx)

        assert requests[0].headers["Authorization"] == "Bearer t-123"
        assert result.success is True
        assert result.response_headers is not None

    asyncio.run(_run())


def test_run_test_execution_skips_failed_dependencies():
    async def _run():
        request_count = {"count": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            request_count["count"] += 1
            return httpx.Response(500, json={"error": "nope"})

        transport = httpx.MockTransport(handler)
        test_plan = APIPlan(
            summary="failing plan",
            base_url="https://api.test",
            steps=[
                APIStep(
                    id="first",
                    description="first call fails",
                    endpoint="/fail",
                    method="GET",
                    expected_status=200,
                    body_format=BodyFormat.NONE,
                ),
                APIStep(
                    id="second",
                    description="should be skipped",
                    endpoint="/next",
                    method="GET",
                    expected_status=200,
                    body_format=BodyFormat.NONE,
                    depends_on=["first"],
                ),
            ],
        )
        test_data = GeneratedTestData(
            payloads=[
                MockedPayload(step_id="first", request_body=None),
                MockedPayload(step_id="second", request_body=None),
            ]
        )

        async with httpx.AsyncClient(transport=transport) as client:
            result = await run_test_execution_phase(
                test_plan, test_data, http_client=client
            )

        assert request_count["count"] == 1  # second request should not fire
        assert result.failed == 1
        assert result.skipped == 1
        assert result.results[1].error == "Skipped due to failed dependencies"

    asyncio.run(_run())


def test_execute_step_reports_exception_type():
    async def _run():
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("socket timed out")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            ctx = ExecutionContext(http_client=client, base_url="https://api.test")
            step = APIStep(
                id="timeout",
                description="simulate timeout",
                endpoint="/slow",
                method="GET",
                expected_status=200,
                body_format=BodyFormat.NONE,
            )
            payload = MockedPayload(step_id="timeout", request_body=None)

            result = await execute_step(step, payload, ctx)

        assert result.success is False
        assert "ReadTimeout" in (result.error or "")

    asyncio.run(_run())


def test_run_test_execution_reports_missing_dependency():
    async def _run():
        test_plan = APIPlan(
            summary="missing dep",
            base_url="https://api.test",
            steps=[
                APIStep(
                    id="orphan",
                    description="depends on missing step",
                    endpoint="/orphan",
                    method="GET",
                    expected_status=200,
                    body_format=BodyFormat.NONE,
                    depends_on=["missing-step"],
                )
            ],
        )
        test_data = GeneratedTestData(
            payloads=[
                MockedPayload(step_id="orphan", request_body=None),
            ]
        )

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        async with httpx.AsyncClient(transport=transport) as client:
            result = await run_test_execution_phase(
                test_plan, test_data, http_client=client
            )

        assert result.failed == 1
        assert "Missing dependencies" in (result.results[0].error or "")

    asyncio.run(_run())
