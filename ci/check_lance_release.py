#!/usr/bin/env python3

"""Determine whether a newer Lance tag exists and expose results for CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import tomllib

LANCE_REPO = "lance-format/lance"

SEMVER_RE = re.compile(
    r"^\s*(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+[0-9A-Za-z.-]+)?\s*$"
)


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[int | str, ...]

    def __lt__(self, other: "SemVer") -> bool:
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (
                other.major,
                other.minor,
                other.patch,
            )
        if self.prerelease == other.prerelease:
            return False
        if not self.prerelease:
            return False  # release > prerelease
        if not other.prerelease:
            return True
        for left, right in zip(self.prerelease, other.prerelease, strict=False):
            if left == right:
                continue
            if isinstance(left, int) and isinstance(right, int):
                return left < right
            if isinstance(left, int):
                return True
            if isinstance(right, int):
                return False
            return str(left) < str(right)
        return len(self.prerelease) < len(other.prerelease)


def parse_semver(raw: str) -> SemVer:
    match = SEMVER_RE.match(raw)
    if not match:
        raise ValueError(f"Unsupported version format: {raw}")
    prerelease = match.group("prerelease")
    parts: tuple[int | str, ...] = ()
    if prerelease:
        parsed: list[int | str] = []
        for piece in prerelease.split("."):
            parsed.append(int(piece) if piece.isdigit() else piece)
        parts = tuple(parsed)
    return SemVer(
        major=int(match.group("major")),
        minor=int(match.group("minor")),
        patch=int(match.group("patch")),
        prerelease=parts,
    )


@dataclass(frozen=True)
class TagInfo:
    tag: str  # e.g. v1.0.0-beta.2
    version: str  # e.g. 1.0.0-beta.2
    semver: SemVer


def run_command(cmd: Sequence[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def fetch_remote_tags() -> list[TagInfo]:
    output = run_command(
        [
            "gh",
            "api",
            "-X",
            "GET",
            f"repos/{LANCE_REPO}/git/refs/tags",
            "--paginate",
            "--jq",
            ".[].ref",
        ]
    )
    tags: list[TagInfo] = []
    for line in output.splitlines():
        ref = line.strip()
        if not ref.startswith("refs/tags/v"):
            continue
        tag = ref.split("refs/tags/")[-1]
        version = tag.removeprefix("v")
        try:
            tags.append(TagInfo(tag=tag, version=version, semver=parse_semver(version)))
        except ValueError:
            continue
    if not tags:
        raise RuntimeError("No Lance tags could be parsed from GitHub API output")
    return tags


def normalize_cargo_version(raw: str) -> str:
    trimmed = raw.strip()
    for prefix in ("^", "~", "="):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix) :].strip()
    if " " in trimmed or "," in trimmed or "|" in trimmed:
        raise ValueError(f"Unsupported Cargo version requirement: {raw}")
    return trimmed


def read_current_version(repo_root: Path) -> str:
    """Read dependencies.lance.version from Cargo.toml."""
    cargo_toml_path = repo_root / "Cargo.toml"
    with cargo_toml_path.open("rb") as handle:
        cargo_toml = tomllib.load(handle)

    deps = cargo_toml.get("dependencies", {})
    spec = deps.get("lance")
    if spec is None:
        raise RuntimeError("Failed to locate [dependencies].lance in Cargo.toml")

    if isinstance(spec, str):
        return normalize_cargo_version(spec)
    if isinstance(spec, dict):
        version = spec.get("version")
        if not version or not isinstance(version, str):
            raise RuntimeError("Failed to locate [dependencies].lance.version in Cargo.toml")
        return normalize_cargo_version(version)

    raise RuntimeError(f"Unsupported [dependencies].lance format: {type(spec)!r}")


def determine_latest_tag(tags: Iterable[TagInfo]) -> TagInfo:
    return max(tags, key=lambda tag: tag.semver)


def write_outputs(args: argparse.Namespace, payload: dict) -> None:
    target = getattr(args, "github_output", None)
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        for key, value in payload.items():
            handle.write(f"{key}={value}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Path to the lance-duckdb repository root",
    )
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Optional file path for writing GitHub Action outputs",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    current_version = read_current_version(repo_root)
    current_semver = parse_semver(current_version)

    tags = fetch_remote_tags()
    latest = determine_latest_tag(tags)
    needs_update = latest.semver > current_semver

    payload = {
        "current_version": current_version,
        "current_tag": f"v{current_version}",
        "latest_version": latest.version,
        "latest_tag": latest.tag,
        "needs_update": "true" if needs_update else "false",
    }

    print(json.dumps(payload))
    write_outputs(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

