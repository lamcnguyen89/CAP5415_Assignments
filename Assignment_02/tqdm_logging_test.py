import logging

logging.basicConfig(level=logging.INFO, 
                    filename="log.txt", 
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s"
                    )

# logging.debug("debug")
# logging.info("info")
# logging.info("Dragon Master Flex")
# logging.warning("warning")
# logging.error("error")
# logging.critical("critical")


# # How to log a variable

# message = "Satanic Master of the Purple Flame"

# logging.info(f"My name is {message}")


# # How to log a stack trace

# try: 
#     1 /0

# except ZeroDivisionError as e:
#     logging.warning("ZeroDivisionError", exc_info=True)
#     logging.exception("ZeroDivisionError")

# Custom Logger

logger = logging.getLogger(__name__) # Convention is to use __name__ and to have a log for each python module
# Gives you the name of the python module

logger.info("test the custom logger")

# Configure the logger to save to different log files using a handler
handler = logging.FileHandler('test.log')
# Setup formatter and configure the handler with the formatter and add to the logger
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Configure formatter with handler
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.info("test the custom logger")
