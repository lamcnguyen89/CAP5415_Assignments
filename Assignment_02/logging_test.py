import logging
 
 
# def main() -> None:
#     # You can control the level of the logging displayed. debug is the lowest level and critical is the highest level
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         filename="basic_log.txt" # Save log to a file
#     )

    
#     logging.debug("This is a debug message")
#     logging.info("This is an info message")
#     logging.warning("This is a warning message")
#     logging.error("This is an error message")
#     logging.critical("This is a critical message")

import datetime
import logging
import time

from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging


logger = logging.getLogger(__name__)


def main():
    fmt = f"%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, filename="basic_log.txt")

    # Set the log level to all tqdm-logging progress bars.
    # Defaults to info - no need to set if you do not want to change the level
    tqdm_logging.set_level(logging.INFO)
    
    # Set the rate how often we update logs
    # Defaults to 10 seconds - optional
    tqdm_logging.set_log_rate(datetime.timedelta(seconds=10))    

    logger.info("This is an INFO test message using Python logging")

    with tqdm(total=60_000, desc="Sample progress", unit_scale=True) as progress_bar:
        for i in range(60_000):
            progress_bar.update(1000)

            # Test postfix output
            progress_bar.set_postfix({"Current time": datetime.datetime.utcnow()})

            time.sleep(0.5)

if __name__ == "__main__":
    main()