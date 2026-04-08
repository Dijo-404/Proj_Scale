from pathlib import Path
import sys

# Ensure tests can import both top-level modules (models.py, server/) and
# the package form (Proj_Scale.*) regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
