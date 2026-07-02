from researchdoc.ingest import DocumentIngestor
from researchdoc.indexer import VectorIndexer
from researchdoc.config import DATA_DIR
from researchdoc.retriever import HybridRetriever

def Ingest():
    DI = DocumentIngestor()
    VI = VectorIndexer()
    
    nodes = DI.process(DATA_DIR)
    index = VI.create_index(nodes)

    hybrid_retriever = HybridRetriever(index=index, similarity_top_K=3)

    query = " SVM is a supervised machine learning algorithm based on statistical learning theory to solve binary/multiclass classification "

    retrieved_nodes = hybrid_retriever.retrieve(query)

    print(f"Total Unique Nodes Retrieved: {len(retrieved_nodes)}\n")

    for i, node in enumerate(retrieved_nodes):
        print(f"--- Node {i+1} (ID: {node.node.node_id}) ---")
        print(node.node.get_content()[:250] + "...\n")

def load():
    VI = VectorIndexer()
    index = VI.load_index()
   
    hybrid_retriever = HybridRetriever(index=index, similarity_top_K=3)

    query = "SVM is a supervised machine learning algorithm..."
    retrieved_nodes = hybrid_retriever.retrieve(query)

    print(f"Total Unique Nodes Retrieved: {len(retrieved_nodes)}\n")
    for i, node in enumerate(retrieved_nodes):
        print(f"--- Node {i+1} (ID: {node.node.node_id}) ---")
        print(node.node.get_content()[:300] + "...\n")
        
if __name__ == "__main__":
    load()