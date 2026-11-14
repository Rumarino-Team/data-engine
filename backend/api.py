import logging
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.factory import create_app

# The server's own INFO messages (device report, interactive mask results) are otherwise dropped.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

app = create_app()
