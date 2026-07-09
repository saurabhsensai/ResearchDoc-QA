import os 
import json 
import numpy as np
from llama_index.llms.groq import Groq
from .config import GROQ_API


class RAGEvaluator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.judge_llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API)
        
    def score_faithfulness(self, context: str, answer: str) -> float:
        prompt = (
            "You are an expert factual judge evaluating an AI system's response for hallucinations.\n"
            "Analyze the given statement and context chunk closely. Break down the statement into individual parts "
            "and verify if every part is completely derived from and supported by the context.\n\n"
            f"Context Chunks:\n{context}\n\n"
            f"AI System Answer:\n{answer}\n\n"
            "Provide your final score as a single raw floating point value between 0.0 and 1.0 (where 1.0 represents "
            "complete verification with absolute backing, and 0.0 represents complete hallucination or zero context coverage).\n"
            "Output ONLY the raw float value, do not include reasoning, intro prose, or formatting."
        )
        
        try: 
            response = self.judge_llm.complete(prompt)
            score = float(str(response).strip())
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.0
    
    def score_relevance(self, query: str, answer: str) -> float:
        prompt = (
            "You are an expert semantic judge evaluating an AI system's response for direct answer relevance.\n"
            "Determine how well the answer directly answers the query question, checks context matching alignment, "
            "and ensures information conciseness without unnecessary rambling or off-topic additions.\n\n"
            f"User Technical Query:\n{query}\n\n"
            f"AI System Answer:\n{answer}\n\n"
            "Provide your final score as a single raw floating point value between 0.0 and 1.0 (where 1.0 represents "
            "a perfect, high-fidelity answer addressing the core question, and 0.0 means completely irrelevant).\n"
            "Output ONLY the raw float value, do not include reasoning, intro prose, or formatting."
        )
        
        try:
            response = self.judge_llm.complete(prompt)
            score = float(str(response).strip())
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.0
        
    def run_benchmark(self, evaluation_set_path: str):
        if not os.path.exists(evaluation_set_path):
            print(f"Error: Evaluation path file {evaluation_set_path} contain no file")
            return
        
        with open(evaluation_set_path, 'r') as f:
            triplets = json.load(f)
        
        faithfulness_score = []
        relevance_score = []
        
        print(f"\n--- Starting RAG Pipeline Benchmark (Total Evaluation Triplets: {len(triplets)}) ---")
        
        for idx, entry in enumerate(triplets):
            query_str = entry["query"]
            ground_truth = entry["ground_truth"]
            
            print(f"\n[{idx + 1}/{len(triplets)}] Query: {query_str}")
            
            nodes = self.rag.retriever.retrieve(query_str)
            formatted_context = "\n\n".join([f"Node Content: {n.get_content()}" for n in nodes])
            
            generated_answer = self.rag.query(query_str)
            print(f"-> Answer: {generated_answer}")
            
            faith_score = self.score_faithfulness(formatted_context, generated_answer)
            rel_score = self.score_relevance(query_str, generated_answer)
            
            print(f"-> Scores - Faithfulness: {faith_score:.2f} | Relevance: {rel_score:.2f}")
            
            faithfulness_score.append(faith_score)
            relevance_score.append(rel_score)
        
        
        print("\n" + "="*50)
        print(" FINAL RAG BENCHMARK METRIC PERFORMANCE")
        print("="*50)
        print(f"Mean System Faithfulness Score : {np.mean(faithfulness_score):.4f}")
        print(f"Mean System Answer Relevance Score: {np.mean(relevance_score):.4f}")
        print("="*50 + "\n")
    
    
            
            
            
            
        
        
    
    
        