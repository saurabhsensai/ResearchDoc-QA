import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ingest import DocumentIngestor


class VectorIndexer:
    def __init__(self, db_path="./chroma_db", collection_name="research_papers"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
        
    def create_index(self, nodes):
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        return index


if __name__ == "__main__":
    DI = DocumentIngestor()
    VI = VectorIndexer()
    
    nodes = DI.process("data/")
    index = VI.create_index(nodes)
    # 1. Fetch all the data directly from the Chroma collection
collection_data = VI.chroma_collection.get()

# 2. Safety check: Did we actually pass in nodes?
if not collection_data['ids']:
    print("ChromaDB is empty! Make sure your 'nodes' list actually has data before running create_index(nodes).")
else:
    # 3. Loop through the data ChromaDB saved to your hard drive
    for i in range(len(collection_data['ids'])):
        print(f"--- Node ID: {collection_data['ids'][i]} ---")
        # Chroma stores the raw text inside the 'documents' list
        print(f"Text Snippet: {collection_data['documents'][i][:200]}...") 
        print(f"Metadata: {collection_data['metadatas'][i]}\n")