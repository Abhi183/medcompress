"""Root conftest.py for MedCompress test suite."""
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent))
