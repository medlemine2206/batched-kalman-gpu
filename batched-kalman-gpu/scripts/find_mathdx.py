"""Print the root of a pip-installed MathDx (pip install nvidia-mathdx), or exit 1."""
import importlib.util
import os
import sys

spec = importlib.util.find_spec("nvidia")
if spec is None or not spec.submodule_search_locations:
    sys.exit(1)
for base in spec.submodule_search_locations:
    root = os.path.join(base, "mathdx")
    if os.path.isfile(os.path.join(root, "include", "cublasdx.hpp")):
        print(root)
        sys.exit(0)
sys.exit(1)
