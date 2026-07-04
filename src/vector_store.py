"""
vector_store.py
================
This module is responsible for two things:
  1. Turning text chunks into "embeddings" (lists of numbers).
  2. Storing/retrieving those embeddings using ChromaDB.

WHAT IS AN EMBEDDING?
-----------------------
An embedding model reads a piece of text and outputs a vector — a fixed-
length list of floating point numbers (for Gemini's embedding-001, that's
768 numbers). This vector is a mathematical "coordinate" in a very
high-dimensional space, positioned so that texts with SIMILAR MEANING end
up at NEARBY coordinates, and texts with different meaning end up far
apart — even if they don't share any of the same words.

For example, "How many vacation days do employees get?" and "What is our
annual leave policy?" would produce two vectors that are close together in
this space, because the model learned during its own training that these
sentences mean nearly the same thing — even though they share almost no
words in common. This is what makes semantic search fundamentally more
powerful than keyword search (which would need the exact word "vacation"
to appear in the document).

WHAT IS ChromaDB?
-------------------
ChromaDB is a "vector database" — a database purpose-built to store these
number-vectors and answer the question "which stored vectors are closest
(most similar) to THIS new vector?" very quickly, even across millions of
entries. Under the hood it uses distance math (by default, cosine
similarity/distance) to rank stored vectors by closeness to a query vector.

We use Chroma in "persistent" mode, meaning it saves its data to a folder
on disk (config.CHROMA_PERSIST_DIR) so your uploaded documents remain
searchable even after you restart the app — you don't need to re-upload
and re-embed everything every time.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    GOOGLE_API_KEY,
    TOP_K_RESULTS,
)


def get_embedding_model():
    """
    Create and return the embedding model object.

    WHY A SEPARATE FUNCTION?
    We need the exact same embedding model both when WRITING new chunks
    into ChromaDB and when READING (querying) it later. The query text and
    the stored chunks must be embedded with the identical model, or their
    vectors won't live in a comparable coordinate space. Centralizing this
    in one function guarantees consistency.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )


def get_vector_store():
    """
    Connect to (or create) the persistent Chroma collection on disk.

    Returns
    -------
    Chroma
        A LangChain Chroma vector store object, already wired up with our
        embedding model and pointed at our on-disk persistence folder.

    NOTE: Calling this function does NOT re-embed anything. It just opens
    a connection to whatever is already saved in CHROMA_PERSIST_DIR (or
    creates an empty collection if nothing exists yet).
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def add_chunks_to_store(chunks):
    """
    Embed a list of text chunks and save them into ChromaDB.

    Parameters
    ----------
    chunks : list[Document]
        Small text chunks produced by text_splitter.py.

    HOW IT WORKS STEP BY STEP
    ---------------------------
    1. We open the persistent vector store (get_vector_store()).
    2. We call .add_documents(chunks). Behind the scenes, LangChain's
       Chroma wrapper does this for every chunk:
         a. Sends chunk.page_content to the Gemini embedding API.
         b. Receives back a 768-number vector.
         c. Stores (vector, chunk.page_content, chunk.metadata) as one
            row inside the Chroma collection on disk.
    3. Chroma automatically generates a unique ID for each stored chunk,
       and persists everything to CHROMA_PERSIST_DIR immediately.

    Returns
    -------
    int
        The number of chunks successfully added — used by the UI to show
        a success message like "Added 42 chunks from 3 documents."
    """
    if not chunks:
        return 0

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return len(chunks)


def similarity_search(query: str, k: int = TOP_K_RESULTS):
    """
    Given a user's natural-language question, return the most semantically
    similar chunks stored in ChromaDB.

    Parameters
    ----------
    query : str
        The user's question, e.g. "What is our data retention policy?"
    k : int
        How many top matches to return (default comes from config.py).

    Returns
    -------
    list[Document]
        The k most similar chunks, each still carrying its original
        metadata (source filename, page number) for citation purposes.

    HOW SIMILARITY SEARCH WORKS
    ------------------------------
    1. The query string is embedded using the SAME embedding model used
       during ingestion — producing one query vector.
    2. Chroma compares this query vector against every stored chunk
       vector using a distance metric (cosine similarity by default) and
       ranks all chunks from most to least similar.
    3. The top `k` chunks (smallest distance / highest similarity) are
       returned.

    This is what lets the app answer "how many sick days do I get" by
    finding a chunk that says "employees accrue 10 days of paid sick leave
    per year" — even though the two sentences share almost no exact words.
    """
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return results


def get_document_count():
    """
    Return how many chunks currently exist in the vector store.
    Used by the Streamlit sidebar to show "X chunks indexed."
    """
    vector_store = get_vector_store()
    # Chroma's underlying client exposes .get() which returns all stored
    # items (ids, documents, metadatas). We just need the count.
    collection_data = vector_store.get()
    return len(collection_data["ids"])


def list_unique_sources():
    """
    Return the list of unique source filenames currently indexed.
    Used by the "Manage Documents" panel in the Streamlit UI.
    """
    vector_store = get_vector_store()
    collection_data = vector_store.get()
    sources = set()
    for metadata in collection_data["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])
    return sorted(sources)


def delete_document(source_filename: str):
    """
    Delete every chunk belonging to a given source filename.

    Parameters
    ----------
    source_filename : str
        The filename as stored in metadata["source"], e.g. "policy.pdf".

    HOW IT WORKS
    ------------
    Chroma lets us delete-by-metadata-filter using the `where` argument:
    `where={"source": source_filename}` means "delete every row whose
    metadata['source'] equals this filename." This is metadata filtering
    in action — the same mechanism could be used to filter searches too
    (e.g. "only search within HR documents").
    """
    vector_store = get_vector_store()
    vector_store._collection.delete(where={"source": source_filename})
