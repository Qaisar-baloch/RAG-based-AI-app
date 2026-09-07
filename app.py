import hashlib
import io
import os

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
TOP_K = 5


# ============================================================
# Load API Key from Streamlit Secrets
# ============================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it under Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ============================================================
# Initialize Groq Client
# ============================================================

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# Load Embedding Model
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# ============================================================
# PDF Text Extraction
# ============================================================

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from every page of the uploaded PDF.
    Returns a list of dictionaries containing page number and text.
    """

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


# ============================================================
# Token-Based Chunking
# ============================================================

def chunk_text(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks based on tokenizer tokens.
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    chunks = []

    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))

        chunk_token_ids = token_ids[start:end]

        chunk = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(token_ids):
            break

        start = end - overlap

    return chunks


# ============================================================
# Create Chunks with Page Metadata
# ============================================================

def create_chunks(pages):
    """
    Create chunks while preserving the PDF page number.
    """

    tokenizer = embedding_model.tokenizer

    all_chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = chunk_text(
            page_text,
            tokenizer,
        )

        for chunk_number, chunk in enumerate(chunks, start=1):
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )

    return all_chunks


# ============================================================
# Create FAISS Vector Index
# ============================================================

def create_faiss_index(chunks):
    """
    Generate embeddings for chunks and store them in FAISS.
    """

    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# Search Relevant Chunks
# ============================================================

def search_documents(question, index, chunks, top_k=TOP_K):
    """
    Embed the question and retrieve the most relevant chunks.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = question_embedding.astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        result = chunks[index_position].copy()
        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# Generate Answer with Groq
# ============================================================

def generate_answer(question, retrieved_chunks):

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information contained
in the provided document context.

Rules:
1. Do not invent information.
2. If the answer is not present in the context, clearly say:
   "I could not find the answer in the uploaded document."
3. Keep the answer clear and concise.
4. When possible, mention the relevant page number.
"""

    user_prompt = f"""
Document context:

{context}

User question:

{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ============================================================
# Session State
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📚 PDF RAG")

    st.markdown(
        """
        Upload a PDF and ask questions about its contents.

        **Pipeline**

        PDF → Text Extraction → Chunking →
        Embeddings → FAISS → Retrieval → Groq
        """
    )

    st.divider()

    st.write("**LLM:**")
    st.code(GROQ_MODEL)

    st.write("**Embedding Model:**")
    st.code(EMBEDDING_MODEL)

    st.write("**Vector Database:**")
    st.code("FAISS")

    st.divider()

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# Main Application
# ============================================================

st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload a PDF document and ask questions about its contents."
)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)


# ============================================================
# Process Uploaded PDF
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_file_hash = hashlib.md5(pdf_bytes).hexdigest()

    # Only process the PDF when a new file is uploaded
    if current_file_hash != st.session_state.file_hash:

        with st.spinner(
            "Processing PDF and creating vector index..."
        ):

            try:

                pages = extract_text_from_pdf(pdf_bytes)

                if not pages:

                    st.error(
                        "No extractable text was found in this PDF. "
                        "It may be a scanned/image-based PDF. "
                        "OCR will be required for such PDFs."
                    )

                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:

                    st.error(
                        "No text chunks could be created from the PDF."
                    )

                    st.stop()

                index = create_faiss_index(chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.file_hash = current_file_hash
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = []

            except Exception as e:

                st.error(
                    f"An error occurred while processing the PDF: {e}"
                )

                st.stop()


# ============================================================
# Show Document Information
# ============================================================

if st.session_state.index is not None:

    st.success(
        f"✅ **{st.session_state.file_name}** is ready. "
        f"Created {len(st.session_state.chunks)} chunks."
    )

    st.divider()

    # ========================================================
    # Display Chat History
    # ========================================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
            ):

                with st.expander("🔎 Retrieved Sources"):

                    for source in message["sources"]:

                        st.markdown(
                            f"**Page {source['page']}** "
                            f"(similarity: {source['score']:.3f})"
                        )

                        st.write(source["text"])

                        st.divider()

    # ========================================================
    # Chat Input
    # ========================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve relevant chunks
        with st.spinner("Searching the document..."):

            retrieved_chunks = search_documents(
                question,
                st.session_state.index,
                st.session_state.chunks,
            )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                try:

                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                    )

                    st.markdown(answer)

                    with st.expander("🔎 Retrieved Sources"):

                        for source in retrieved_chunks:

                            st.markdown(
                                f"**Page {source['page']}** "
                                f"(similarity: {source['score']:.3f})"
                            )

                            st.write(source["text"])

                            st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks,
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"Sorry, an error occurred while "
                        f"generating the answer: {e}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )

else:

    st.info(
        "👆 Upload a PDF above to start chatting with your document."
    )
