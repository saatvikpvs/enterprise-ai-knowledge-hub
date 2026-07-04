# 📚 Enterprise AI Knowledge Hub (RAG)

An enterprise knowledge assistant that answers natural-language questions
over your own documents (PDF, DOCX, TXT, Markdown, HTML) using
Retrieval-Augmented Generation (RAG): it retrieves the most relevant
passages from a ChromaDB vector store and asks Gemini to answer using only
that retrieved context — with source citations for every answer.

# Live Demo

The application has been successfully deployed on Streamlit Community Cloud.

## Live Application:

https://enterprise-ai-knowledge-app-yqczd3zvonwis9gsjlzfh7.streamlit.app/

You can directly open the above link in any web browser to try the application without installing anything.

### How to test
Open the application using the link above.
Upload one or more supported documents:
  PDF
  DOCX
  TXT
  Markdown
  HTML
  Click Process Documents.
  Ask questions about the uploaded documents.
The application retrieves relevant information using Retrieval-Augmented Generation (RAG) and displays the answer along with the supporting source citations.
 
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

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/saatvikpvs/enterprise-ai-knowledge-hub
cd enterprise-ai-knowledge-hub
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file in the project root and add:

```text
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open the application in your browser:

```
http://localhost:8501
```

The application is now ready to use. Upload supported documents (PDF, DOCX, TXT, Markdown, or HTML), process them, and ask natural-language questions about their contents.
---

##  Ask questions

Type a question in the chat box at the bottom, for example:

- "How many days of paid sick leave do employees get?"
- "What is the maximum payload of the AR-200 robot?"
- "How often must vendor access be reviewed?"

Expand **"📎 Sources"** under each answer to see exactly which
document/page the answer came from.

##  Test the application

Good smoke tests using the included sample documents:

| Question | Expected source |
|---|---|
| "What is the notice period for a manager?" | hr_leave_policy.txt |
| "How long until data is deleted after a customer relationship ends?" | it_security_policy.md |
| "What temperature range can the AR-200 operate in?" | product_faq.html |
| "What is the capital of France?" | Should say it can't find this in the uploaded documents |

That last test confirms the grounding/anti-hallucination behavior is
working — the app should refuse to answer from general knowledge.


## Adding more document types later

To support a new format (say, `.csv` or `.pptx`):

1. Install the relevant LangChain community loader (e.g.
   `CSVLoader`, `UnstructuredPowerPointLoader`).
2. In `src/document_loader.py`, add an `elif extension == ".csv":` branch
   inside `load_single_document()` that instantiates the new loader.
3. Add the extension to `SUPPORTED_EXTENSIONS` in `config.py` and to the
   `type=[...]` list in `app.py`'s `st.file_uploader`.

Because every other module works with the generic LangChain `Document`
object (not format-specific logic), no other file needs to change.

## Improving retrieval quality

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
