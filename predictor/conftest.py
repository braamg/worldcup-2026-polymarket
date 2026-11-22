"""Make the src/ package importable from the project root during pytest runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
