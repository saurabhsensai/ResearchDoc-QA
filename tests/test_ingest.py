import pytest
from researchdoc.ingest import DocumentIngestor

def test_missing_directory_raises_error():
    ingestor = DocumentIngestor()
    with pytest.raises(FileNotFoundError):
        ingestor.process("./directory_that_does_not_exist")