import os
import sys
import importlib

# Ensure src/ is on sys.path so imports work during collection
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    mod = importlib.import_module("causomic.data_analysis.gene_set")
    # Prevent pytest from treating functions inside this module as tests
    setattr(mod, "__test__", False)
except Exception:
    # If import fails during collection, ignore; tests will import it later
    pass
