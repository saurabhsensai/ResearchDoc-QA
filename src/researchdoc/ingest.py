import os 
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

class DocumentIngestor: 
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = SentenceSplitter(chunk_size= self.chunk_size, 
                                       chunk_overlap=self.chunk_overlap)
        
    def load_doc(self, data_dir: str):
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found")

        print(f"loading documents from {data_dir} .....")
        documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
        print(f"Loaded {len(documents)} document pages.")
        return documents
    
    def chunk_documents(self, documemnts): 
        nodes = self.parser.get_nodes_from_documents(documemnts)
        print(f"Created {len(nodes)} chunks (nodes).")
        return nodes
    
    def process(self, data_dir: str): 
        documents = self.load_doc(data_dir)
        return self.chunk_documents(documents)


if __name__ == "__main__":
    ingestor = DocumentIngestor()
    nodes = ingestor.process("./data")
    
    if nodes: 
        print("\n-----sample chuck-------")
        print(nodes[0].text)
        
        
        
        
        
        

