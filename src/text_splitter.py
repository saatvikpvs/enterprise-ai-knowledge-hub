"""
text_splitter.py
=================
This module takes the raw Document objects produced by document_loader.py
(which might each contain a whole page or a whole file of text) and breaks
them into smaller, retrieval-friendly "chunks."

WHY CHUNKING IS NECESSARY
--------------------------
Imagine a 40-page employee handbook. If we embedded the ENTIRE handbook as
one single vector, a search for "how many sick days do I get" would return
"the whole handbook" — technically correct, but useless, because:
  1. It's too large to fit in the LLM's context window alongside other
     retrieved documents.
  2. The embedding of a 40-page document is a "blurry average" of many
     unrelated topics (benefits, dress code, security policy, etc.), so it
     won't score highly for similarity against a specific, narrow question.

By splitting into small chunks (a paragraph or two at a time), each chunk's
embedding represents ONE specific idea, so semantic search can pinpoint the
exact paragraph about sick days instead of returning an entire document.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """
    Split a list of Document objects into smaller chunks.

    Parameters
    ----------
    documents : list[Document]
        The raw Documents returned by document_loader.py.

    Returns
    -------
    list[Document]
        A longer list of smaller Document chunks. Metadata (like the
        source filename and page number) is automatically copied onto
        every chunk that came from that Document, so citations still work
        after splitting.

    HOW RecursiveCharacterTextSplitter WORKS
    ------------------------------------------
    "Recursive" here means: it tries to split on the most meaningful
    boundary first, and only falls back to cruder boundaries if a chunk is
    still too big. By default its separator list (in priority order) is:

        1. "\\n\\n"  (blank line -> paragraph break)
        2. "\\n"    (single newline -> line break)
        3. " "     (space -> word break)
        4. ""      (character -> absolute last resort)

    So it FIRST tries to cut text at paragraph boundaries. If a single
    paragraph is still longer than CHUNK_SIZE, it then tries to cut at
    line breaks within that paragraph. If a single line is still too long,
    it cuts at word boundaries. This means chunks almost never end mid-word
    or mid-sentence unless the text has no natural breaks at all (e.g. one
    giant unbroken block of text).

    chunk_size and chunk_overlap come from config.py so the whole app tunes
    from one place (see config.py for why 1000/150 were chosen).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # length_function tells the splitter how to "measure" a chunk.
        # We use plain character count (len) rather than token count for
        # simplicity; this is a reasonable approximation for most models.
        length_function=len,
    )

    # split_documents() (a method LangChain provides on the splitter) takes
    # a list of Document objects, splits doc.page_content for each one, and
    # automatically clones doc.metadata onto every resulting chunk.
    chunks = splitter.split_documents(documents)

    # We add a per-chunk index to metadata. This is purely cosmetic/
    # diagnostic — it lets us show "Chunk 3 of policy.pdf" in the UI if we
    # ever want finer-grained citations than just the filename.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
