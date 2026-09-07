import io
import os
from typing import List, Dict

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

# Chunking is token-based using the embedding model's tokenizer.
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
TOP_K = 5


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


# -----------------------------
# PDF + RAG functions
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes) -> List[Dict]:
    """Extract text page-by-page from a PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_text(text: str, tokenizer, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping token-based chunks."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return []

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(token_ids):
            break

        start = end - overlap

    return chunks


def build_chunks(pages: List[Dict], tokenizer) -> List[Dict]:
    """Create chunks while preserving source page numbers."""
    all_chunks = []

    for page in pages:
        chunks = chunk_text(page["text"], tokenizer)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page": page["page"],
            })

    return all_chunks


def build_faiss_index(chunks: List[Dict], embedding_model):
    """Create a normalized cosine-similarity FAISS index."""
    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(query: str, index, chunks: List[Dict], embedding_model, top_k: int = TOP_K):
    """Retrieve the most relevant chunks for a user query."""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def generate_answer(query: str, retrieved_chunks: List[Dict], client: Groq) -> str:
    """Ask the Groq open-weight model to answer using retrieved context."""
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.
If the answer cannot be found in the context, clearly say that the information
is not available in the uploaded document.

Do not invent facts. Keep the answer clear and useful.
When possible, mention the relevant PDF page number(s).
"""

    user_prompt = f"""Document context:

{context}

User question:
{query}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Chatbot")
st.caption("Upload a PDF, index it with FAISS, and ask questions using an open-weight Groq model.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL_NAME}`")
    st.write(f"**Chunk size:** {CHUNK_SIZE} tokens")
    st.write(f"**Chunk overlap:** {CHUNK_OVERLAP} tokens")
    st.write(f"**Top-K retrieval:** {TOP_K}")

    if st.button("🗑️ Clear current document"):
        for key in ["index", "chunks", "file_name", "num_pages"]:
            st.session_state.pop(key, None)
        st.rerun()

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The PDF is processed in memory for the current Streamlit session.",
)

if uploaded_file is not None:
    # Only rebuild the index when a new file is uploaded.
    if st.session_state.get("file_name") != uploaded_file.name:
        pdf_bytes = uploaded_file.getvalue()

        with st.spinner("Extracting PDF text..."):
            pages = extract_pdf_text(pdf_bytes)

        if not pages:
            st.error(
                "No extractable text was found. This app currently works best with "
                "text-based PDFs. Scanned PDFs need OCR."
            )
            st.stop()

        embedding_model = load_embedding_model()

        with st.spinner("Creating token-based chunks..."):
            chunks = build_chunks(pages, embedding_model.tokenizer)

        if not chunks:
            st.error("Could not create text chunks from this PDF.")
            st.stop()

        with st.spinner("Creating embeddings and FAISS index..."):
            index = build_faiss_index(chunks, embedding_model)

        st.session_state["index"] = index
        st.session_state["chunks"] = chunks
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["num_pages"] = len(pages)

        # Reset conversation when a new document is indexed.
        st.session_state["messages"] = []

        st.success(
            f"Indexed **{uploaded_file.name}** — {len(pages)} pages, "
            f"{len(chunks)} chunks."
        )

if "index" not in st.session_state:
    st.info("👆 Upload a PDF to build your searchable FAISS index.")
    st.stop()

st.success(
    f"Current document: **{st.session_state['file_name']}** "
    f"({st.session_state['num_pages']} pages)"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about your PDF...")

if query:
    st.session_state["messages"].append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        with st.chat_message("assistant"):
            st.error(
                "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
                "or set it as an environment variable."
            )
        st.stop()

    embedding_model = load_embedding_model()

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            results = retrieve(
                query,
                st.session_state["index"],
                st.session_state["chunks"],
                embedding_model,
            )

        if not results:
            answer = "I couldn't find relevant information in the document."
            st.markdown(answer)
        else:
            try:
                client = get_groq_client(api_key)

                with st.spinner("Generating answer..."):
                    answer = generate_answer(query, results, client)

                st.markdown(answer)

                with st.expander("🔎 Retrieved sources"):
                    for i, item in enumerate(results, start=1):
                        st.markdown(
                            f"**Source {i} — PDF page {item['page']} "
                            f"(similarity: {item['score']:.3f})**"
                        )
                        st.write(item["text"])

            except Exception as exc:
                answer = f"Sorry, the Groq request failed: `{exc}`"
                st.error(answer)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
    })
