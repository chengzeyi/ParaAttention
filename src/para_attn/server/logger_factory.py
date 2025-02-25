import logging


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] " "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
        )

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_
