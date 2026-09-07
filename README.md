# 📚 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their contents.

The application extracts text from the PDF, splits it into token-based chunks, creates embeddings, stores them in a FAISS vector index, retrieves the most relevant chunks, and uses Groq's `openai/gpt-oss-120b` model to generate an answer.

## 🚀 Live Demo

Deployed using Streamlit Community Cloud.

> Add your Streamlit Cloud URL here after deployment.

---

## 🧠 How It Works

The application follows this RAG pipeline:

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Token-Based Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Top-K Relevant Chunks
    ↓
Groq LLM
    ↓
Final Answer
