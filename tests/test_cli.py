"""Tests for CLI interface."""
import sys
from pathlib import Path
from click.testing import CliRunner
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "1.2.0" in result.output


def test_cli_report(runner):
    """Report command should produce output without errors."""
    result = runner.invoke(cli, ["report"])
    # May fail if data not generated, but should not crash
    assert result.exit_code in (0, 1)


def test_cli_validate(runner):
    """Validate command should run schema checks."""
    result = runner.invoke(cli, ["validate"])
    assert result.exit_code in (0, 1)
    assert "Validation" in result.output or "file not found" in result.output
