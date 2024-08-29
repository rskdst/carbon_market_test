import logging
import logging.config
import os

import yaml

with open("src/logging_config.yaml", "r") as file:
    config = yaml.safe_load(file)
    for handler in config["handlers"].values():
        filename = handler.get("filename")
        if filename:
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
    logging.config.dictConfig(config)


def setup_logging(logger_name):
    return logging.getLogger(logger_name)
