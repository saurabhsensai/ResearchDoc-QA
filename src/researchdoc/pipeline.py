import os 
from .config import GROQ_API
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate

class RAGPipeline:
    def __init__(self, retriever, llm_model_name='llama-3.1-8b-instant'):
        self.retriever = retriever
        self.llm = Groq(model=llm_model_name,
                        api_key=GROQ_API)
        self.system_prompt = PromptTemplate(
            "You are a strict, factual AI assistant. You must answer the user's query ONLY using the provided context chunks.\n"
            "If the answer cannot be found in the context, you must exactly say: 'I cannot find the answer in the provided documents.'\n"
            "You must cite your sources inline using the metadata provided with each chunk (e.g., [Source: filename, Page: X]).\n\n"
            "Context:\n"
            "{context_str}\n\n"
            "Query:\n"
            "{query_str}\n\n"
            "Answer:"
        )
        
    def query(self, query_str: str) -> str:
        
        nodes = self.retriever.retrieve(query_str)
            
        context_chunks = []
        for node in nodes: 
            source_file = node.metadata.get("file_name")
            page_num = node.metadata.get("page_label")
                
            text = node.get_content().strip()
                
            formatted_chunk = f"Text: {text}\nMetadata: [Source: {source_file}, Page: {page_num}]"
            context_chunks.append(formatted_chunk)
                
        context_str = "\n\n---\n\n".join(context_chunks)
            
        prompt = self.system_prompt.format(
            context_str=context_str, 
            query_str=query_str
            )
            
        response = self.llm.complete(prompt)
        
        return str(response)
                
        
        
    