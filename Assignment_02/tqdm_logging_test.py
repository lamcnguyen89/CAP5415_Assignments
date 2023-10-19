from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
import datetime
import logging
import time
import io

# Class for logging progress of training
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
           # tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record) 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.addHandler(TqdmLoggingHandler())

