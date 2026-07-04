"""
qa_chain.py
===========
This is the "brain" of the application. It ties together:
  - retrieval (pulling relevant chunks from ChromaDB via vector_store.py)
  - the LLM (Gemini)
  - conversational memory (so follow-up questions like "what about part-time
    staff?" still make sense)
  - prompt engineering (instructing the LLM HOW to use the retrieved text)
  - citation extraction (so the UI can show "Sources: policy.pdf")

WHY WE BUILD THE CHAIN MANUALLY (RATHER THAN A ONE-LINE HELPER)
------------------------------------------------------------------
LangChain offers pre-built chains like `ConversationalRetrievalChain` that
hide all of these steps. We instead wire the pieces together explicitly
using LangChain Expression Language (LCEL, the `|` pipe syntax) because:
  1. It makes every step visible and easy to explain (you can literally
     read the pipeline top to bottom).
  2. It makes it trivial to see exactly what context and history get sent
     to the LLM, and to grab the source documents separately for citation
     display — which prebuilt chains often make awkward.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from config import LLM_MODEL_NAME, LLM_TEMPERATURE, GOOGLE_API_KEY
from src.vector_store import similarity_search


# ---------------------------------------------------------------------------
# THE PROMPT TEMPLATE
# ---------------------------------------------------------------------------
# This is the actual instruction we send to the LLM every single time,
# with placeholders ({context}, {question}, {chat_history}) filled in per
# request. This is "prompt engineering": we are explicitly telling the
# model:
#   - its role ("enterprise knowledge assistant")
#   - what its ONLY source of truth is (the retrieved context)
#   - what to do if the answer isn't in the context (say so, don't guess)
#   - to consider prior conversation turns for follow-up questions
#
# This grounding instruction is the single biggest lever for reducing
# hallucination: by explicitly telling the model "if it's not in the
# context, say you don't know," we discourage it from falling back on its
# own possibly-wrong general knowledge.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an enterprise knowledge assistant. Answer the user's "
     "question using ONLY the information in the CONTEXT below. "
     "If the answer cannot be found in the context, clearly say "
     "\"I could not find this information in the uploaded documents\" "
     "instead of guessing or using outside knowledge. "
     "Be concise and factual. When useful, quote key figures or terms "
     "directly from the context.\n\n"
     "CONTEXT:\n{context}"),
    # MessagesPlaceholder inserts the running conversation history here,
    # so the model can resolve pronouns/follow-ups like "what about them?"
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_context(chunks):
    """
    Combine a list of retrieved chunks into a single text block to inject
    into the prompt, with a clear separator and source label per chunk.

    WHY LABEL EACH CHUNK WITH ITS SOURCE HERE?
    Labelling inside the context (not just in metadata) lets the LLM
    itself reference "According to policy.pdf..." naturally in its answer
    if it chooses to, and it makes debugging easier — when you print the
    context, you can immediately tell where each piece of text came from.
    """
    labeled_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown source")
        page = chunk.metadata.get("page")
        page_info = f", page {page + 1}" if page is not None else ""
        labeled_chunks.append(
            f"[Excerpt {i} — from {source}{page_info}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(labeled_chunks)


def get_llm():
    """
    Create the Gemini chat model object used to generate answers.

    temperature comes from config.py — we keep it low (0.2) so the model
    stays grounded and factual rather than creative, which matters for an
    enterprise fact-lookup tool.
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


def convert_history_to_messages(chat_history):
    """
    Convert our simple internal history format (a list of
    {"role": "user"/"assistant", "content": "..."} dicts, which Streamlit's
    session_state stores) into LangChain's HumanMessage/AIMessage objects,
    which the prompt template and the model expect.

    WHY KEEP TWO FORMATS?
    Streamlit session_state works best with plain dicts (they're easy to
    render as chat bubbles). LangChain's chat model API wants its own
    Message classes. This converter is the small "adapter" between the UI
    layer and the LangChain layer.
    """
    messages = []
    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def answer_question(question: str, chat_history: list):
    """
    The main entry point: given a user's question and the conversation so
    far, retrieve relevant chunks, ask the LLM, and return both the answer
    text and the list of source chunks used (for citation display).

    Parameters
    ----------
    question : str
        The user's latest natural-language question.
    chat_history : list[dict]
        Prior turns of the conversation, e.g.
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}]

    Returns
    -------
    dict with keys:
        "answer": str          -> the generated response text
        "sources": list[Document] -> the chunks used, for citation display

    EXECUTION FLOW (this is the heart of RAG)
    --------------------------------------------
    1. RETRIEVE: similarity_search(question) hits ChromaDB and returns the
       top-K most semantically relevant chunks (see vector_store.py).
    2. AUGMENT: format_context() glues those chunks into one labeled text
       block, and we build the full prompt (system instructions + context
       + prior chat turns + the new question).
    3. GENERATE: the prompt is sent to Gemini via the LCEL chain
       (prompt | llm | StrOutputParser()), which returns the final natural
       -language answer string.
    "Retrieval-Augmented Generation" is literally steps 1-2-3: retrieve
    context, augment the prompt with it, then generate the answer.
    """
    # STEP 1: RETRIEVE
    retrieved_chunks = similarity_search(question)

    if not retrieved_chunks:
        # No documents indexed yet, or nothing even remotely relevant was
        # found. We short-circuit here instead of calling the LLM with an
        # empty context, since that would just invite a hallucinated
        # answer.
        return {
            "answer": (
                "I don't have any relevant documents to answer this "
                "question yet. Please upload some documents first."
            ),
            "sources": [],
        }

    # STEP 2: AUGMENT
    context_text = format_context(retrieved_chunks)
    history_messages = convert_history_to_messages(chat_history)

    # STEP 3: GENERATE
    # This is LangChain Expression Language (LCEL): the `|` operator pipes
    # the output of one component into the next, like a Unix pipe.
    #   RAG_PROMPT.invoke(...)  -> produces a formatted list of messages
    #   llm.invoke(messages)    -> sends them to Gemini, gets an AIMessage
    #   StrOutputParser()       -> extracts the plain string from AIMessage
    chain = RAG_PROMPT | get_llm() | StrOutputParser()

    answer_text = chain.invoke({
        "context": context_text,
        "chat_history": history_messages,
        "question": question,
    })

    return {
        "answer": answer_text,
        "sources": retrieved_chunks,
    }
