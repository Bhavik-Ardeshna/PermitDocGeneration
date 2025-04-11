import logging
import os

# level = 'read from GCP secrets - <Shortened_Service_Name>_LOGGER_LEVEL'
# logger_name = 'read from GCP secrets - <Shortened_Service_Name>_LOGGER_NAME'
# # Either read from secrets in this file (It would read from secrets everytime logger object is called)

# Or during initialization, read from secrets and set as environment variables:
level = os.getenv("ATLAS_LOGGER_LEVEL", "DEBUG")
logger_name = os.getenv("ATLAS_LOGGER_NAME", "atlas-dev-log")


def setup_logger():
    """Function to set up logger with specified settings.
    Returns:
        logging.Logger: The configured logger object.
    """
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(funcName)s %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# setup logger for entire application
logger = setup_logger()
