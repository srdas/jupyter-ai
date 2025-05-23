import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAGServer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG server with an embedding model and FAISS index
        
        :param model_name: Sentence transformer model for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the vector database
        
        :param documents: List of text documents to add
        """
        print(f"Adding {len(documents)} documents to the vector database.")
        # Generate embeddings for documents
        embeddings = self.model.encode(documents)
        
        # Initialize FAISS index if not already created
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index and store original documents
        self.index.add(embeddings)
        self.documents.extend(documents)
        print(f"Added {len(documents)} documents to the vector database.")
        print(f"Total documents in database: {len(self.documents)}")
    
    def query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query the vector database and return most relevant documents
        
        :param query: Search query
        :param top_k: Number of top results to return
        :return: List of dictionaries with document and similarity score
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'document': self.documents[idx],
                'similarity_score': float(1 / (1 + dist))  # Convert to Python float
            })
        
        return results

def create_app():
    """
    Create and configure the Flask application
    
    :return: Configured Flask application
    """
    app = Flask(__name__)
    rag_server = RAGServer()

    @app.route('/add_documents', methods=['POST'])
    def add_documents():
        """
        Endpoint to add documents to the vector database
        """
        documents = request.json.get('documents', [])
        rag_server.add_documents(documents)
        return jsonify({'status': 'success', 'documents_added': len(documents)})

    @app.route('/query', methods=['POST'])
    def query_documents():
        """
        Endpoint to query the vector database
        """
        query = request.json.get('query', '')
        top_k = request.json.get('top_k', 10)
        results = rag_server.query(query, top_k)
        return jsonify({'results': results})

    return app

def run_server(host: str = '127.0.0.1', port: int = 5000):
    """
    Run the RAG server
    
    :param host: Host to bind the server
    :param port: Port to run the server on
    """
    app = create_app()
    app.run(host=host, port=port)

def main():
    """
    Main entry point for running the RAG server
    """
    run_server()

if __name__ == '__main__':
    main()
