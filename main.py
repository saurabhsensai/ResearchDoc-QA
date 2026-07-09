from researchdoc.ingest import DocumentIngestor
from researchdoc.indexer import VectorIndexer
from researchdoc.config import DATA_DIR, EVALUATION_SET
from researchdoc.retriever import HybridRetriever
from researchdoc.pipeline import RAGPipeline
from researchdoc.evaluate import RAGEvaluator


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
    RAG = RAGPipeline(hybrid_retriever)
    
    evaluator = RAGEvaluator(RAG)
    evaluator.run_benchmark(evaluation_set_path=EVALUATION_SET)
    
        
if __name__ == "__main__":
    load()