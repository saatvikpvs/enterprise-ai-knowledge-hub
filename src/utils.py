"""
utils.py
========
Small, generic helper functions that don't belong to any single stage of
the RAG pipeline (loading, splitting, storing, or generating), but are
used by the Streamlit UI layer (app.py).

Keeping these separate from app.py keeps the UI file focused purely on
layout and user interaction, rather than mixing in file-system plumbing.
"""

import os
from config import UPLOAD_DIR, SUPPORTED_EXTENSIONS


def save_uploaded_file(uploaded_file):
    """
    Save a file object coming from Streamlit's file_uploader widget onto
    disk, inside UPLOAD_DIR, so the rest of the pipeline (which expects
    file PATHS, not in-memory file objects) can process it normally.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The object Streamlit gives us for each uploaded file. It behaves
        like a file handle — `.name` gives the original filename, and
        `.getbuffer()` gives its raw bytes.

    Returns
    -------
    str
        The full path where the file was saved.
    """
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    # "wb" = write binary. We write binary (not text) mode because PDFs
    # and DOCX files are binary formats, not plain text.
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def is_supported_file(filename: str) -> bool:
    """Check whether a filename has one of our supported extensions."""
    _, extension = os.path.splitext(filename)
    return extension.lower() in SUPPORTED_EXTENSIONS


def format_source_label(chunk):
    """
    Build a short, human-readable citation label for a single retrieved
    chunk, e.g. "policy.pdf (page 3)" or "handbook.docx".

    Used by app.py when rendering the "Sources" section under each answer.
    """
    source = chunk.metadata.get("source", "Unknown source")
    page = chunk.metadata.get("page")
    if page is not None:
        return f"{source} (page {page + 1})"
    return source
