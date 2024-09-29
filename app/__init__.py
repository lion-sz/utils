# Ensure that the paths are correctly loaded.

import sys
import pathlib

utils_path = pathlib.Path(__file__).parents[1].absolute()
if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))


from app._app import app, state, format_float  # noqa: F401 E402
