"""A tiny synthetic Qwen3.8-shaped model pair (BF16 base + ModelOpt-NVFP4 baseline) for tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bittrellis.synthetic import TINY_TEXT, make_tiny  # noqa: F401  (TINY_TEXT re-exported for tests)


@pytest.fixture(scope="session")
def tiny_all(tmp_path_factory) -> tuple[Path, Path, Path]:
    return make_tiny(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def tiny_models(tiny_all) -> tuple[Path, Path]:
    return tiny_all[0], tiny_all[1]
