import asyncio
import httpx
import logfire
from pathlib import Path
import argparse
from devtools import debug
import re
from datetime import datetime

from src.agent import agent, AgentDeps

# Configure logfire
logfire.configure()


def get_next_run_id(output_dir: Path) -> int:
    if not output_dir.exists():
        return 1

    max_id = 0
    for file in output_dir.glob("*_test_plan.json"):
        try:
            # Expecting format: NNN_url_timestamp_test_plan.json
            parts = file.name.split("_")
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
    parser = argparse.ArgumentParser(description="Generate a REST API test plan.")
    parser.add_argument("openapi_url", help="URL to the OpenAPI specification")
    parser.add_argument(
        "--repo", help="Path to the local repository for analysis", default=None
    )
    args = parser.parse_args()

    repo_path = Path(args.repo) if args.repo else None

    print(f"Starting analysis for: {args.openapi_url}")
    if repo_path:
        print(f"Analyzing repository at: {repo_path}")

    async with httpx.AsyncClient() as client:
        deps = AgentDeps(http_client=client, repo_path=repo_path)

        prompt = f"Create a test plan for the API at {args.openapi_url}."
        if repo_path:
            prompt += " Use the provided tools to analyze the repository code for better context."

        result = await agent.run(prompt, deps=deps)

        print("\n--- Generated Test Plan ---")
        debug(result.output)

        # Output directory setup
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        run_directory = create_run_directory(output_dir, args.openapi_url)

        test_plan_filename = run_directory / "test_plan.json"

        with open(test_plan_filename, "w") as f:
            f.write(result.output.model_dump_json(indent=2))
        print(f"\nTest plan saved to {test_plan_filename}")


if __name__ == "__main__":
    asyncio.run(main())
