from pathlib import Path
import sys


def setup_project_paths() -> Path:
    """
    Add the repository root and vendored dependencies to sys.path.

    This allows scripts to run without requiring editable installation of
    the local MedMAP package or vendored dp-transformers.
    """
    repo_root = Path(__file__).resolve().parents[1]

    paths = [
        repo_root,
        repo_root / "library" / "dp-transformers" / "src",
    ]

    for path in paths:
        if not path.exists():
            raise RuntimeError(f"Required path does not exist: {path}")

        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return repo_root