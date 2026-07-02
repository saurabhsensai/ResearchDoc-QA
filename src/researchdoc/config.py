from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT/"data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DOCSTORE_DIR = PROJECT_ROOT / "docstore"

