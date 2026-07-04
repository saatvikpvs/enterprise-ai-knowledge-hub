# 📚 Enterprise AI Knowledge Hub (RAG)

An enterprise knowledge assistant that answers natural-language questions
over your own documents (PDF, DOCX, TXT, Markdown, HTML) using
Retrieval-Augmented Generation (RAG): it retrieves the most relevant
passages from a ChromaDB vector store and asks Gemini to answer using only
that retrieved context — with source citations for every answer.

## Folder structure

```
enterprise-ai-knowledge-hub/
├── app.py                     # Streamlit UI — the app's front door
├── config.py                  # All tunable settings in one place
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── document_loader.py     # PDF/DOCX/TXT/MD/HTML -> plain text
│   ├── text_splitter.py       # plain text -> small retrieval chunks
│   ├── vector_store.py        # embeddings + ChromaDB read/write
│   ├── qa_chain.py            # retrieval + prompt + LLM + memory
│   └── utils.py                # small UI-layer helper functions
├── data/
│   ├── sample_docs/            # ready-to-use sample enterprise documents
│   ├── uploaded_docs/          # files uploaded via the UI land here
│   └── chroma_db/               # persisted vector database (auto-created)
└── diagrams/
    ├── architecture.svg
    └── rag_pipeline.svg
```

## How it works (one paragraph)

When you upload a document, it's parsed into plain text, split into
overlapping ~1000-character chunks, converted into vector embeddings by
Gemini's embedding model, and stored in ChromaDB. When you ask a question,
your question is embedded with the same model, ChromaDB returns the most
semantically similar chunks, and those chunks are inserted into a prompt
that instructs Gemini to answer **only** from that context — reducing
hallucination and letting every answer cite its source document.

See `diagrams/architecture.svg` and `diagrams/rag_pipeline.svg` for visual
walkthroughs, and read the docstring at the top of every file in `src/`
for a detailed, beginner-friendly explanation of what it does and why.

---

## 1. Software to install

- **Python 3.10 or 3.11** (recommended — `unstructured` and `chromadb`
  can have issues on very new Python versions like 3.13; 3.10/3.11 is the
  safest choice).
- **Git** (optional, only if you want to version-control the project).
- A code editor such as VS Code.

Check your Python version:

```bash
python --version
```

## 2. Create a virtual environment

A virtual environment keeps this project's dependencies isolated from
other Python projects on your machine.

```bash
cd enterprise-ai-knowledge-hub
python -m venv venv
```

## 3. Activate the virtual environment

**Windows (PowerShell):**
```bash
venv\Scripts\activate
```

**Windows (cmd.exe):**
```bash
venv\Scripts\activate.bat
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

You'll know it worked because your terminal prompt will now start with
`(venv)`.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, LangChain, ChromaDB, the Gemini integration, and
the document-parsing libraries (pypdf, docx2txt, unstructured).

> If `unstructured` fails to install on Windows, install the "Build Tools
> for Visual Studio" (C++ build tools) first, then re-run pip install.

## 5. Get a Gemini API key

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with a Google account.
3. Click "Create API key" and copy the generated key.

Gemini currently offers a free tier that's sufficient for testing this
project.

## 6. Create the `.env` file

Copy the example file:

```bash
# Mac/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

## 7. Configure environment variables

Open `.env` in your editor and paste your key:

```
GOOGLE_API_KEY=AIzaSy...your_real_key...
```

Save the file. `config.py` automatically loads this at startup via
`python-dotenv` — you never need to type your key into any code file.

## 8. ChromaDB setup

No separate server needed — this project uses ChromaDB in **embedded /
persistent mode**, meaning it runs inside the same Python process as the
Streamlit app and simply saves its data to `data/chroma_db/` on disk.
There is nothing extra to start or configure.

## 9. Launch the application

```bash
streamlit run app.py
```

This opens your browser automatically at `http://localhost:8501`. If it
doesn't, open that URL manually.

## 10. Upload documents

- Use the sidebar's **"Upload documents"** widget.
- Try the ready-made sample files in `data/sample_docs/`:
  `hr_leave_policy.txt`, `it_security_policy.md`, `product_faq.html`.
- Click **"Process Documents"** and wait for the success message.

## 11. Ask questions

Type a question in the chat box at the bottom, for example:

- "How many days of paid sick leave do employees get?"
- "What is the maximum payload of the AR-200 robot?"
- "How often must vendor access be reviewed?"

Expand **"📎 Sources"** under each answer to see exactly which
document/page the answer came from.

## 12. Test the application

Good smoke tests using the included sample documents:

| Question | Expected source |
|---|---|
| "What is the notice period for a manager?" | hr_leave_policy.txt |
| "How long until data is deleted after a customer relationship ends?" | it_security_policy.md |
| "What temperature range can the AR-200 operate in?" | product_faq.html |
| "What is the capital of France?" | Should say it can't find this in the uploaded documents |

That last test confirms the grounding/anti-hallucination behavior is
working — the app should refuse to answer from general knowledge.

## 13. Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `GOOGLE_API_KEY is not set` (shown in UI) | `.env` missing or empty | Re-check step 6/7 |
| `google.api_core.exceptions.PermissionDenied` | Invalid/expired API key | Generate a new key at AI Studio |
| `ModuleNotFoundError: No module named 'langchain_google_genai'` | Dependencies not installed / wrong venv active | Re-run `pip install -r requirements.txt` inside the activated venv |
| Upload succeeds but answers say "I could not find this information" for something that IS in the document | Chunk didn't retrieve top-K, or file failed to parse | Check the "Processing log" in the sidebar for a load error; try re-uploading |
| `unstructured` install fails on Windows | Missing C++ build tools | Install "Build Tools for Visual Studio", then retry |
| Streamlit opens but the page is blank/errors on first load | An exception occurred before rendering | Check the terminal running `streamlit run app.py` for the Python traceback |

## 14. Adding more document types later

To support a new format (say, `.csv` or `.pptx`):

1. Install the relevant LangChain community loader (e.g.
   `CSVLoader`, `UnstructuredPowerPointLoader`).
2. In `src/document_loader.py`, add an `elif extension == ".csv":` branch
   inside `load_single_document()` that instantiates the new loader.
3. Add the extension to `SUPPORTED_EXTENSIONS` in `config.py` and to the
   `type=[...]` list in `app.py`'s `st.file_uploader`.

Because every other module works with the generic LangChain `Document`
object (not format-specific logic), no other file needs to change.

## 15. Improving retrieval quality

- **Tune chunk size/overlap** in `config.py` — smaller chunks (e.g. 500)
  give more precise citations; larger chunks (e.g. 1500) preserve more
  surrounding context per match.
- **Increase `TOP_K_RESULTS`** in `config.py` if answers seem to be
  missing relevant information spread across several chunks.
- **Add metadata filtering** — e.g. tag documents by department at upload
  time and filter `similarity_search()` to only search within a given
  department for more focused results.
- **Add a re-ranking step** — retrieve a larger candidate set (e.g. 15
  chunks) then use a cross-encoder or the LLM itself to re-rank and keep
  only the best 4 before generating the answer.
- **Hybrid search** — combine semantic (vector) search with traditional
  keyword search (BM25) for queries that include exact identifiers like
  product codes or ticket numbers, which embeddings alone can struggle
  to match precisely.

## 16. Deploying the application

- **Streamlit Community Cloud** (simplest): push this repo to GitHub,
  connect it at https://share.streamlit.io, and add `GOOGLE_API_KEY` as a
  "secret" in the app settings instead of committing a `.env` file.
- **Docker**: write a `Dockerfile` that copies the project, runs
  `pip install -r requirements.txt`, and starts with
  `CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]`,
  then deploy the container to any cloud provider (Render, Railway, AWS
  ECS, etc.).
- For production/enterprise use, consider swapping the local persisted
  ChromaDB for a hosted vector database (e.g. Chroma Cloud, Pinecone, or
  pgvector) so multiple app instances can share one knowledge base.

---

## Concepts explained

**Retrieval-Augmented Generation (RAG):** instead of asking an LLM to
answer purely from what it memorized during training, RAG first retrieves
relevant, up-to-date, or private text (your own documents), then feeds
that text to the LLM as context for generation. This grounds the answer
in verifiable source material rather than the model's static, general
knowledge — and lets the model answer accurately about documents it has
never seen before.

**Embeddings:** a numeric "fingerprint" of meaning. An embedding model
converts a piece of text into a fixed-length vector of numbers positioned
so that texts with similar meaning sit close together in that
multi-dimensional space, even if they don't share the same words.

**ChromaDB / vector database:** a database specialized in storing these
vectors and quickly answering "which stored vectors are closest to this
new one?" — the mathematical basis of semantic search.

**Chunking:** splitting long documents into small, self-contained pieces
before embedding them, so that each embedding represents one focused idea
instead of a blurry average of an entire document.

**Prompt engineering:** carefully wording the instructions sent to the
LLM (see `RAG_PROMPT` in `src/qa_chain.py`) so it uses the retrieved
context correctly, cites it appropriately, and refuses to answer when the
context doesn't contain the answer.

**Conversational memory:** the running chat history is fed back into the
prompt on every turn, so the model can resolve follow-up questions like
"what about part-time employees?" by referring to what was discussed
previously.

Every module in `src/` has an extensive docstring at the top explaining
its specific responsibility in even more depth — start there if you want
to go deeper on any one piece.
