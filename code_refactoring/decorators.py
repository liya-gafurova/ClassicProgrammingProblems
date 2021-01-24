from functools import wraps
import logging


def create_logger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    sh = logging.StreamHandler()
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    sh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(sh)
    return logger


def logged(logger):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """

    def decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                # log the exception
                err_msg = f"There was an exception in  {function.__name__}. Error message: {e.args}"
                logger.exception(err_msg)

        return wrapper

    return decorator
