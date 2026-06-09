"""Make the repo root importable so `from src import ...` works in tests no matter how
pytest is invoked. The bare `pytest` console script (used in CI) does not add the
current directory to sys.path, unlike `python -m pytest`. A root conftest.py is the
version-independent fix; pytest.ini's pythonpath setting is the belt-and-suspenders.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
