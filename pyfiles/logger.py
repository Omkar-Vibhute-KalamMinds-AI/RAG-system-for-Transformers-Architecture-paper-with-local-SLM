 # src/logger.py
import logging
import os
from datetime import datetime

LOG_DIR = "LLMOps/RAGOpsLogings" 
os.makedirs(LOG_DIR, exist_ok=True)  

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    )

logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
    )

logger = logging.getLogger('__main__')

