import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from .ingest import DocumentIngestor
from .config import DATA_DIR, CHROMA_DIR, DOCSTORE_DIR

class VectorIndexer:
    def __init__(self, db_path=CHROMA_DIR, collection_name="research_papers"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
        
    def create_index(self, nodes):
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(vector_store=vector_store,docstore=docstore)
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        storage_context.persist(persist_dir=str(DOCSTORE_DIR))
        return index
    
    def load_index(self):
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(
            persist_dir=str(DOCSTORE_DIR),
            vector_store=vector_store
        )

        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
      
        return index

