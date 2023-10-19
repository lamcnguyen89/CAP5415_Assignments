import time
from tqdm_logger import TqdmLogger
import tqdm

log_file = 'temp_log.log'
tqdm_stream = TqdmLogger(log_file)

tqdm_stream.reset()

for x in tqdm(range(100), file = tqdm_stream):
    time.sleep(.5)