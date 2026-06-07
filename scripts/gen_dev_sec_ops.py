"""Generate dev_sec_ops.yml from the live git history and project metadata.

Run from the repository root right before the final commit:

    python scripts/gen_dev_sec_ops.py

The DockerHub image name is read from the DOCKERHUB_USERNAME environment
variable so that no real account handle is hardcoded in the repository.
"""

import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "dev_sec_ops.yml"


def _git(*args: str) -> str:
    """Run a git command in the repository and return trimmed stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_last_commits(count: int = 5) -> list[str]:
    """Return full SHA-1 hashes of the last ``count`` commits."""
    raw = _git("log", f"-{count}", "--format=%H")
    return [line for line in raw.splitlines() if line]


def get_current_branch() -> str:
    """Return the current branch name."""
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def read_coverage() -> str:
    """Read total coverage percentage from coverage.xml if it exists."""
    coverage_file = BASE_DIR / "coverage.xml"
    if not coverage_file.exists():
        return "not_measured"
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(coverage_file).getroot()
        line_rate = float(root.attrib.get("line-rate", "0"))
        return f"{round(line_rate * 100, 2)}%"
    except (ET.ParseError, ValueError):
        return "not_measured"


def build_document() -> str:
    """Assemble the dev_sec_ops.yml content."""
    commits = get_last_commits()
    branch = get_current_branch()
    image_name = os.getenv("DOCKERHUB_USERNAME", "your_dockerhub_username")
    coverage = read_coverage()

    commit_lines = "\n".join(f'    - "{commit}"' for commit in commits)

    return f"""project:
  name: boston-housing-mlops
  repository_branch: {branch}

docker:
  image_name: {image_name}/boston-housing-api
  image_tag: latest
  signing: not_enabled

git:
  last_5_commits:
{commit_lines}

quality:
  test_framework: pytest
  test_command: "pytest -v"
  coverage: {coverage}

security:
  secrets_in_repo: false
  dependency_audit: not_enabled
  image_scan: not_enabled

artifacts:
  model_path: models/model.joblib
  metrics_path: metrics.json
  scenario_path: scenario.json
"""


def main() -> None:
    OUTPUT_PATH.write_text(build_document(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
