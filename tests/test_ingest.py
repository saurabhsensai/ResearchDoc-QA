import pytest
from src import ingest

def test_missing_directory_raises_error():
    ingestor = ingest.DocumentIngestor()
    with pytest.raises(FileNotFoundError):
        ingestor.process("./directory_that_does_not_exist")