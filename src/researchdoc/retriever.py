from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle

class HybridRetriever:
    def __init__(self, index, similarity_top_K=20, rerank_top_n=5):
        #vector retriever
        self.vector_retriever = index.as_retriever(similarity_top_k=similarity_top_K)
        
        docstore = index.storage_context.docstore
        nodes = list(docstore.docs.values())

        if not nodes:
            raise ValueError(
                "index.docstore.docs is empty."
                "Make sure documents were inserted into the index before creating HybridRetriever."
            )
    
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=similarity_top_K
        )
        
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
            top_n=rerank_top_n
        )
    
    def retrieve(self, query_str: str):
        vector_nodes = self.vector_retriever.retrieve(query_str)
        bm25_nodes = self.bm25_retriever.retrieve(query_str)
        
        combined_nodes = []
        seen_ids = set()
        
        for node_with_score in vector_nodes + bm25_nodes:
            node_id = node_with_score.node.node_id
            
            if node_id not in seen_ids:
                combined_nodes.append(node_with_score)
                seen_ids.add(node_id)
                
        query_bundle = QueryBundle(query_str)
        reranked_nodes = self.reranker.postprocess_nodes(
            combined_nodes, 
            query_bundle=query_bundle
        )
        
        return reranked_nodes
    
    
    