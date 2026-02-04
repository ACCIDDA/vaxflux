"""Validate release readiness and create a GitHub release."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
INIT_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'^__version__\s*=\s*"(?P<version>\d+\.\d+\.\d+)"\s*$',
    re.MULTILINE,
)
GH_RELEASE_LIST_TYPE: Final[TypeAdapter[list[dict[Literal["tagName"], str]]]] = (
    TypeAdapter(list[dict[Literal["tagName"], str]])
)


@dataclass(frozen=True, order=True)
class SemVer:
    """Semantic version container."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> SemVer:
        """Parse a semantic version string."""
        match = SEMVER_PATTERN.fullmatch(raw)
        if match is None:
            msg = f"Invalid semantic version: {raw!r}."
            raise ValueError(msg)
        major, minor, patch = (int(part) for part in match.groups())
        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        """Format semantic version."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_next_increment_from(self, previous: SemVer) -> bool:
        """Return whether this version is the next semantic increment."""
        is_patch = (
            self.major == previous.major
            and self.minor == previous.minor
            and self.patch == previous.patch + 1
        )
        is_minor = (
            self.major == previous.major
            and self.minor == previous.minor + 1
            and self.patch == 0
        )
        is_major = (
            self.major == previous.major + 1 and self.minor == 0 and self.patch == 0
        )
        return is_patch or is_minor or is_major


def get_gh_bin() -> str:
    """Resolve the GitHub CLI path or raise if unavailable."""
    gh_bin = shutil.which("gh")
    if gh_bin is None:
        msg = "GitHub CLI (`gh`) not found on PATH."
        raise FileNotFoundError(msg)
    return gh_bin


def get_versions() -> SemVer:
    """Read and validate project versions from pyproject and __init__."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pyproject_version = str(pyproject["project"]["version"])
    init_text = (REPO_ROOT / "src" / "vaxflux" / "__init__.py").read_text()
    init_match = INIT_VERSION_PATTERN.search(init_text)
    if init_match is None:
        msg = "Could not find __version__ in src/vaxflux/__init__.py."
        raise ValueError(msg)
    init_version = init_match.group("version")
    if pyproject_version != init_version:
        msg = (
            "Version mismatch between pyproject.toml and src/vaxflux/__init__.py: "
            f"{pyproject_version} != {init_version}."
        )
        raise ValueError(msg)
    return SemVer.parse(pyproject_version)


def validate_and_extract_changelog_section(
    version: SemVer, *, create: bool = False
) -> str:
    """Validate changelog format and return release notes for the current version."""
    changelog_lines = (REPO_ROOT / "CHANGELOG.md").read_text().splitlines()

    h2_headings = [line for line in changelog_lines if line.startswith("## ")]
    if not h2_headings:
        msg = "CHANGELOG.md must contain at least one level-2 release heading."
        raise ValueError(msg)

    if any(h.lower() == "## [unreleased]" for h in h2_headings):
        msg = "CHANGELOG.md must not contain a `## [Unreleased]` heading."
        raise ValueError(msg)

    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    expected_heading = f"## [{version}] - {today}"
    if h2_headings[0] != expected_heading:
        msg = (
            "Top CHANGELOG.md heading must match current version and today's date: "
            f"`{expected_heading}`."
        )
        raise ValueError(msg)

    header_idx = changelog_lines.index(expected_heading)
    section_lines: list[str] = []
    for line in changelog_lines[header_idx + 1 :]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if stripped.startswith("-") and re.search(r"[A-Za-z0-9]", stripped) is None:
            msg = f"Invalid changelog bullet point: {line!r}."
            raise ValueError(msg)
        section_lines.append(line)
    notes = "\n".join(section_lines).strip()
    if not notes:
        msg = f"CHANGELOG section for {version} is empty."
        raise ValueError(msg)
    if not create:
        sys.stdout.write(
            f"Extracted release notes for version {version}:\n\n{notes}\n\n"
        )

    return notes


def validate_release_state(version: SemVer) -> None:
    """Validate version state against existing GitHub releases."""
    proc = subprocess.run(  # noqa: S603
        [
            get_gh_bin(),
            "release",
            "list",
            "--json",
            "tagName",
            "--limit",
            "1",
            "--order",
            "desc",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    releases = GH_RELEASE_LIST_TYPE.validate_json(proc.stdout)
    if not releases:
        return
    latest_tag = str(releases[0]["tagName"])
    latest_version_raw = latest_tag.removeprefix("v")
    latest_version = SemVer.parse(latest_version_raw)
    if version == latest_version:
        msg = f"Version {version} already exists as a GitHub release ({latest_tag})."
        raise ValueError(msg)
    if not version.is_next_increment_from(latest_version):
        msg = (
            "Current version must be the next semantic increment from the latest "
            f"release. current={version}, latest={latest_version}."
        )
        raise ValueError(msg)


def create_release(version: SemVer, notes: str, *, create: bool = False) -> None:
    """Create GitHub release using gh CLI."""
    tag = f"v{version}"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        prefix="vaxflux-release-notes-",
    ) as temp_file:
        temp_file.write(notes)
        notes_path = temp_file.name
        command = [
            get_gh_bin(),
            "release",
            "create",
            tag,
            "--title",
            tag,
            "--notes-file",
            notes_path,
        ]
        if version.major == 0:
            command.append("--prerelease")
        if create:
            subprocess.run(command, check=True)  # noqa: S603
            return
        sys.stdout.write(f"Dry run of GitHub release command: {' '.join(command)}")


def main() -> None:
    """Run release checks and optionally create the release."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the GitHub release after all checks pass.",
    )
    args = parser.parse_args()

    version = get_versions()
    validate_release_state(version)
    notes = validate_and_extract_changelog_section(version, create=args.create)
    create_release(version, notes, create=args.create)


if __name__ == "__main__":
    main()
