from pathlib import Path

import pytest

from enterprise_agent_rl.common import Settings
from enterprise_agent_rl.data import initialize_database


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return initialize_database(tmp_path / "business.sqlite3")


@pytest.fixture
def settings() -> Settings:
    return Settings()
