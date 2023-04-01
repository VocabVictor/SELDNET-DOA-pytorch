from logging import basicConfig, INFO, getLogger, StreamHandler, Formatter
from logging.config import dictConfig
from os.path import join, exists

from Config import Config
from Utils import read_json


class Logger:

    def __init__(self, default_level=INFO):
        """
        Setup logging configuration
        """
        config = Config()
        log_config = config.log_config
        save_dir = config.log_dir
        self.logger = getLogger()
        self.testlog = getLogger("test")
        handler = StreamHandler("test.txt")
        formatter = Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.testlog.addHandler(handler)
        if exists(log_config):
            self.handler = []
            config = read_json(log_config)
            # modify logging paths based on run config
            for _, handler in config['handlers'].items():
                if 'filename' in handler:
                    handler['filename'] = join(save_dir, handler['filename'])
            dictConfig(config)
        else:
            print("Warning: logging configuration file is not found in {}.".format(log_config))
            basicConfig(level=default_level)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def info(self, message):
        self.logger.info(message)

    def critical(self, message):
        self.logger.critical(message)

    def error(self, message):
        self.logger.error(message)
