import sys
import pathlib

utils_path = pathlib.Path(__file__).parents[1].absolute()
if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))

from app import app  # noqa: E402


if __name__ == "__main__":
    app()
