from jupyter_ai.mcp_rag.mcp_client_2 import RAGClient
import pandas as pd

def curate_results(results):
    """
    Curate the results from the RAG server
    :param results: Results from the server
    :return: Curated results
    """
    result = results['results'] # [0]['document']
    df = pd.DataFrame(result).drop_duplicates()
    curated_results = df["document"].str.cat(sep=" ")
    return curated_results

# Initialize client
client = RAGClient()

# Send a document file
client.send_file('/Users/sanjivda/Downloads/learn.txt')
print("File sent successfully.", "\n\n")

# Example 1
results = client.query("What is the main topic of this article about quantum computing?")
print(curate_results(results), "\n\n")

# Example 2
results = client.query("What is the criticism of Microsoft in the article?")
print(curate_results(results), "\n\n")
