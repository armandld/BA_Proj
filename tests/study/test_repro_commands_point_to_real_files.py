"""Every documented executable command must point to a current file."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", ROOT / "CLAUDE.md", *sorted(
    (ROOT / "docs").glob("*.md"))]
COMMAND = re.compile(
    r"(?:^|\s)(?:\.venv/bin/python|python3?|bash)\s+"
    r"((?:src|study|scripts|figures|tests)/[A-Za-z0-9_./-]+\.(?:py|sh))"
)


def documented_targets(path):
    return set(COMMAND.findall(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", DOCS, ids=lambda path: path.name)
def test_documented_commands_point_to_current_files(path):
    missing = sorted(
        target for target in documented_targets(path)
        if not (ROOT / target).is_file())
    assert not missing, f"{path.relative_to(ROOT)}: missing targets {missing}"


def test_command_sweep_is_not_empty():
    found = {(path.name, target) for path in DOCS
             for target in documented_targets(path)}
    assert len(found) >= 10, f"documentation command sweep found only {found}"


@pytest.mark.parametrize("path", DOCS, ids=lambda path: path.name)
def test_live_documentation_has_no_removed_versioned_study_path(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"study/v[234]/", text), path.relative_to(ROOT)


def test_canonical_launchers_are_documented():
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    for target in (
        "scripts/run_rented_campaign.sh",
        "scripts/run_dns_campaign.sh",
        "scripts/run_study_v3.sh",
        "scripts/run_confirmatory_campaign.sh",
    ):
        assert target in text
