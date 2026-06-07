"""Integration test -- requires DEEPSEEK_API_KEY. Run manually."""

import os

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_simple_problem() -> None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    # This is a manual test scaffold -- actual execution requires assembled prompts
    pytest.skip(
        "Integration test requires assembled project -- run via test_e2e.sh"
    )
