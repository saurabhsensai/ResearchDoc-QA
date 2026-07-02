from llama_index.core.storage.docstore import SimpleDocumentStore
from pathlib import Path

DOCSTORE_DIR = "docstore"    # ← change this

docstore = SimpleDocumentStore.from_persist_dir(DOCSTORE_DIR)
print(f"Number of documents in docstore: {len(docstore.docs)}")

if docstore.docs:
    first_node = next(iter(docstore.docs.values()))
    print(f"First node text preview: {first_node.text[:200]}...")