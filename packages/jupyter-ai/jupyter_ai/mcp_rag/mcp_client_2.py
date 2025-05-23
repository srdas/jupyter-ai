import requests
from typing import List, Dict, Any
import os

class RAGClient:
    def __init__(self, server_url: str = 'http://127.0.0.1:5000'):
        """
        Initialize the RAG client
        
        :param server_url: URL of the RAG server
        """
        self.server_url = server_url
    
    def send_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Send documents to the RAG server to be stored in the vector database
        
        :param documents: List of text documents to send
        :return: Server response
        """
        response = requests.post(
            f'{self.server_url}/add_documents', 
            json={'documents': documents}
        )
        return response.json()
    
    def send_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a text file and send its contents to the RAG server
        
        :param file_path: Path to the text file
        :return: Server response
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Split file into document shards (e.g., paragraphs)
            documents = f.read().split('\n\n')
            print(f"Sending {len(documents)} documents to the server.")
        
        return self.send_documents(documents)
    
    def query(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Query the vector database on the RAG server
        
        :param query: Search query
        :param top_k: Number of top results to return
        :return: Query results from the server
        """
        response = requests.post(
            f'{self.server_url}/query', 
            json={'query': query, 'top_k': top_k}
        )
        print(f"Querying server with: {query}")
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        # Check if the response is valid
        if not response.json():
            raise Exception("Empty response from server")
        # Check if the response contains expected keys
        if 'results' not in response.json():
            raise Exception("Invalid response format from server")
        return response.json()

def main():
    """
    Example usage of the RAG client
    """
    client = RAGClient()
    
    # Example: Send a file to the server
    try:
        result = client.send_file('example_document.txt')
        print("Documents added:", result)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example: Query the vector database
    query_result = client.query("What is the main topic?")
    print("Query Results:", query_result)

if __name__ == '__main__':
    main()
