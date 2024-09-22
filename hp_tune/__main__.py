from sys import argv
import pathlib

from config_reader.reader import ConfigReader
from hp_tune.visualizer import build_app

if __name__ == "__main__":
    if "visualize" in argv:
        config = ConfigReader("HP_TUNING").load(pathlib.Path("config"))
        app = build_app(config)
        app.run(debug=True, host="0.0.0.0")
