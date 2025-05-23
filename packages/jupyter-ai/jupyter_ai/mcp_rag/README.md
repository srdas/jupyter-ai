# RAG modules with a server and client implementation. 

The modules are located in:

* /packages/jupyter-ai/jupyter_ai/mcp_rag/mcp_rag.py
* /packages/jupyter-ai/jupyter_ai/mcp_rag/mcp_client.py

Key features:

* Uses FAISS for vector database storage
* Sentence Transformers for document embeddings
* Flask-based server for document storage and querying
* Client can send text files and query the database

To use the modules:

1. Install dependencies: `pip install -r mcp_rag/requirements.txt`
2. Start the server from Terminal:
    - Start `$python`, then from the Python command line: 
    - `>>> from jupyter_ai.mcp_rag import run_server`
    - `>>> run_server()`
3. Use the client to send documents and query, see `mcp_client.py`, you can run `python mcp_test.py`

Follow-up work:

1. Call the RAG from the `SanjivPersona` using custom slash commands for `/learn` and `/ask`. 
2. Replace the simple RAG placeholder with a MCP RAG server and also add a MCP Client. 

