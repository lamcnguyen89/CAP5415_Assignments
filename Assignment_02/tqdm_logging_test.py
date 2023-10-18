import logging
import tqdm

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

import time

logging.basicConfig(level=logging.INFO, filename="log.txt", filemode="w")
log = logging.getLogger(__name__)
log.addHandler(TqdmLoggingHandler())
for i in tqdm.tqdm(range(100)):
    log.info(f"{i}. The Darkness is here")
    time.sleep(0.1)