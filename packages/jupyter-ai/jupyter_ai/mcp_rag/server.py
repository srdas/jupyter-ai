from mcp.server.fastmcp import FastMCP
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import httpx
import sys

# Create a FastMCP instance
mcp = FastMCP("RAG2")

embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
model = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")


@mcp.tool()
async def load_documents(file_path: str) -> str:
    """Load documents from a file."""
    loader = TextLoader(file_path)
    documents = loader.load()
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    # Create a vector store
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="db")
    print(f"Loaded {len(documents)} documents.")
    return vectorstore


@mcp.tool()
async def retrieve_db(file_path: str, query: str) -> str:
    """Use the retrieval chain to get the answer and source documents."""
    vectorstore = await load_documents(file_path)
    qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 3}))
    return qa.run(query)


@mcp.tool()
async def get_url_text(url: str) -> str:
    """Fetch text data from a URL."""
    headers = {
        "User-Agent": "mcp-server/1.0",
        "Accept": "text/plain"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        print(f"Fetched {len(response.text)} characters from {url}", file=sys.stderr)
        return response.text
    except httpx.RequestError as e:
        return f"An error occurred while fetching the URL: {e}"
    

@mcp.tool()
async def retrieve_url(url: str, query: str) -> str:
    """Retrieve text data from the URL and answer the query.
    This function fetches text data from the URL, processes it, and returns the answer to the query.
    Do not use the `retrieve_db` function in this case. It is only used for the `load_documents` function.
    """
    data = await get_url_text(url)
    print(f"Fetched text from {url}", file=sys.stderr)
        
    if "error" in data:
        return data["error"]
    else:
        prompt = f"Answer the following question based on the provided context.\n\nContext:\n{data}\n\nQuestion: {query}\nAnswer:"
        response = model.invoke(prompt)
        return response
 

if __name__ == "__main__":
    # Start the FastMCP server
    mcp.run()
