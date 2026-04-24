from __future__ import annotations

import os
import time
from collections.abc import Callable
from uuid import uuid4

import pytest


@pytest.fixture(scope="session")
def require_integration_enabled() -> None:
    """Skip integration tests unless explicitly enabled."""
    if os.getenv("EVRET_RUN_INTEGRATION") != "1":
        pytest.skip(
            "Integration tests are disabled. Set EVRET_RUN_INTEGRATION=1 to run them."
        )


@pytest.fixture(scope="session")
def require_docker_daemon(require_integration_enabled: None) -> None:
    """Skip integration tests when Docker daemon is not reachable."""
    docker = pytest.importorskip("docker")
    try:
        docker.from_env().ping()
    except Exception as exc:
        pytest.skip(f"Docker daemon is unavailable: {exc}")


def wait_until_ready(check: Callable[[], None], timeout_seconds: float = 30.0) -> None:
    """Retry readiness checks until they pass or timeout."""
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            check()
            return
        except Exception as exc:  # pragma: no cover - only used in integration runtime
            last_error = exc
            time.sleep(0.5)

    if last_error is not None:
        raise TimeoutError("Timed out waiting for service readiness") from last_error
    raise TimeoutError("Timed out waiting for service readiness")


def unique_collection_name(prefix: str) -> str:
    """Generate a short unique collection name for integration tests."""
    return f"{prefix}_{uuid4().hex[:10]}"


@pytest.fixture(scope="session")
def wait_until_ready_helper() -> Callable[[Callable[[], None], float], None]:
    """Fixture exposing readiness retry helper."""
    return wait_until_ready


@pytest.fixture
def unique_collection_name_factory() -> Callable[[str], str]:
    """Fixture exposing unique collection name generator."""
    return unique_collection_name
