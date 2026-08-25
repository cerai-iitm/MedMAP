from pathlib import Path
import sys


def setup_project_paths() -> Path:
    """
    Add the repository root and vendored dependencies to sys.path.

    This allows every entry point under scripts/ to run directly from a
    repository checkout (``python -m scripts.<name>`` from the repo root)
    without requiring ``pip install -e .``. If the medmap package and the
    vendored dp-transformers fork are already installed, the inserted paths
    simply shadow-resolve to the same code.
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
