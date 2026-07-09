from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv()

GROQ_API = os.getenv("groq_api")

DATA_DIR = PROJECT_ROOT/"data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DOCSTORE_DIR = PROJECT_ROOT / "docstore"
EVALUATION_SET = PROJECT_ROOT / "data/evaluation_set.json"

