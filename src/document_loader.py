"""
document_loader.py
===================
This module answers the question: "Given a file on disk, how do I turn it
into plain text that LangChain can work with?"

WHY THIS FILE EXISTS
---------------------
A PDF is a binary format full of layout/font information. A DOCX file is
actually a hidden .zip file full of XML. An HTML file is full of tags. None
of that structural "noise" is useful to an LLM — we just want the human-
readable words. This module's whole job is: format in -> clean text out.

LangChain ships "document loaders" for exactly this purpose. Each loader
returns a list of `Document` objects — a Document is just a simple object
with two attributes:
    - page_content: the actual extracted text (a string)
    - metadata: a dictionary of extra info (like which file it came from,
      which page number, etc.) that travels alongside the text forever.

That metadata is CRITICAL for citations later: when the LLM answers a
question using a chunk, we can say "this came from policy.pdf, page 3"
because the metadata never gets separated from the text.
"""

import os
from langchain_community.document_loaders import (
    PyPDFLoader,        # extracts text from PDF files, page by page
    Docx2txtLoader,      # extracts text from Word .docx files
    TextLoader,          # reads plain .txt / .md files (Markdown is plain text)
    BSHTMLLoader,        # strips HTML tags using BeautifulSoup, keeps visible text
)

# NOTE ON LOADER CHOICES
# We deliberately avoid `Unstructured*Loader` classes (UnstructuredMarkdownLoader
# / UnstructuredHTMLLoader) even though LangChain offers them. Under the hood
# they depend on the `unstructured` library's NLP sentence-tokenizer, which
# tries to download NLTK data from the internet the first time it runs. On a
# machine with a locked-down network (common in corporate/enterprise
# environments — exactly where this app is meant to run!) that download can
# fail and crash ingestion. TextLoader (for Markdown, which is just plain
# text) and BSHTMLLoader (which uses the lightweight `beautifulsoup4` library
# instead of `unstructured`) give the same end result with no hidden network
# dependency.


def load_single_document(file_path: str):
    """
    Load ONE file and return a list of LangChain Document objects.

    Parameters
    ----------
    file_path : str
        Full path to the file on disk (e.g. "data/uploaded_docs/policy.pdf")

    Returns
    -------
    list[Document]
        Each Document holds a chunk of raw extracted text + metadata.
        (Note: at this stage the "chunks" are usually per-page or
        per-file, NOT yet the small chunks we'll create later in
        text_splitter.py. This is just raw extraction.)

    HOW IT WORKS
    ------------
    We look at the file's extension (the part after the last dot) and pick
    the matching LangChain loader class. This is a simple "strategy
    pattern" — one function, many possible strategies, chosen at runtime.
    """
    # os.path.splitext splits "policy.pdf" into ("policy", ".pdf").
    # We lowercase the extension so ".PDF" and ".pdf" both work.
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".pdf":
        # PyPDFLoader creates one Document per PAGE, and automatically
        # stores the page number in metadata["page"]. That's what lets us
        # later cite "Source: policy.pdf, page 4".
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        # Docx2txtLoader extracts all visible text from a Word document as
        # a single Document (Word files don't have a native "page" concept
        # the way PDFs do, since pagination depends on the viewer/printer).
        loader = Docx2txtLoader(file_path)

    elif extension == ".txt":
        # TextLoader just reads the raw text file. encoding="utf-8" avoids
        # errors when files contain special characters (emojis, accents).
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension == ".md":
        # Markdown IS plain text (the "#"/"*" symbols are just lightweight
        # formatting hints), so TextLoader reads it perfectly well. The
        # LLM itself understands Markdown syntax fine when it later reads
        # this text as context, so there's no real need to strip the "#"
        # and "*" characters out.
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension in (".html", ".htm"):
        # BSHTMLLoader uses the `beautifulsoup4` library to parse the HTML
        # DOM and pull out only the human-readable visible text, discarding
        # tags like <div>, <p>, <script>, <style>.
        loader = BSHTMLLoader(file_path)

    else:
        # We deliberately fail loudly instead of silently skipping unknown
        # file types — a silent skip could make a user think a document
        # was ingested when it actually wasn't.
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported types are: .pdf, .docx, .txt, .md, .html"
        )

    # .load() is the method every LangChain loader implements. It does the
    # actual file reading + parsing and returns the list of Document
    # objects described above.
    documents = loader.load()

    # Attach the original filename to every Document's metadata. Some
    # loaders already add a "source" key with the full file path, but we
    # overwrite it with just the filename to keep citations short and
    # readable in the UI (e.g. "policy.pdf" instead of
    # "/home/user/app/data/uploaded_docs/policy.pdf").
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = filename

    return documents


def load_all_documents(folder_path: str):
    """
    Loop over every supported file in a folder and load them all.

    Parameters
    ----------
    folder_path : str
        Folder containing documents to ingest (e.g. the sample dataset
        folder, or the folder where user uploads are saved).

    Returns
    -------
    list[Document]
        A single flat list combining the Documents from every file.

    WHY THIS FUNCTION EXISTS
    -------------------------
    The Streamlit UI and any batch-ingestion script both need to process
    "everything in this folder" rather than one file at a time. Rather
    than duplicating that looping logic in multiple places, we centralize
    it here.
    """
    all_documents = []
    supported_extensions = (".pdf", ".docx", ".txt", ".md", ".html", ".htm")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip directories and unsupported files silently here (this is a
        # bulk "load everything you can" operation, unlike
        # load_single_document which is used for a single explicit upload
        # and should fail loudly on bad input).
        if not os.path.isfile(file_path):
            continue
        if not filename.lower().endswith(supported_extensions):
            continue

        try:
            docs = load_single_document(file_path)
            all_documents.extend(docs)
        except Exception as error:
            # We don't want one broken file to crash the ingestion of an
            # entire folder, so we log the error and keep going.
            print(f"[document_loader] Failed to load {filename}: {error}")

    return all_documents
