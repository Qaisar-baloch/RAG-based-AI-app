# 📚 PDF RAG Chatbot with Streamlit, FAISS & Groq

A beginner-friendly Retrieval-Augmented Generation (RAG) application that lets a user upload a PDF and ask questions about its contents.

The application:

1. Extracts text from the uploaded PDF.
2. Splits the text into overlapping token-based chunks.
3. Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Stores the embeddings in a FAISS vector index.
5. Embeds the user's question and retrieves the most relevant chunks.
6. Sends only the retrieved context to Groq.
7. Uses OpenAI's open-weight `gpt-oss-120b` model through the Groq API to generate the answer.
8. Displays the retrieved source chunks and PDF page numbers.

## 🧠 Architecture

```text
                 ┌─────────────────┐
                 │   User uploads  │
                 │       PDF       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Extract text   │
                 │     (pypdf)     │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Token-based     │
                 │ chunking        │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Sentence        │
                 │ Transformer     │
                 │ embeddings      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ FAISS vector    │
                 │ index            │
                 └────────┬────────┘
                          │
              User question
                          │
                          ▼
                 ┌─────────────────┐
                 │ Embed question  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Top-K FAISS     │
                 │ retrieval       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Groq API        │
                 │ gpt-oss-120b    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Answer + source │
                 │ pages/chunks    │
                 └─────────────────┘
```

## 🛠️ Tech stack

- **Frontend:** Streamlit
- **PDF extraction:** pypdf
- **Tokenization:** tokenizer included with the Sentence Transformer
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database/index:** FAISS CPU
- **LLM inference:** Groq API
- **LLM:** `openai/gpt-oss-120b`
- **Deployment:** Streamlit Community Cloud
- **Source control:** GitHub

## 📁 Project structure

```text
pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## 🚀 Run locally

### 1. Clone your repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a Groq API key

Create an API key in your Groq account.

Do **not** put the real API key inside `app.py`, `README.md`, or GitHub.

### 5. Set the API key locally

You can use an environment variable.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

macOS/Linux:

```bash
export GROQ_API_KEY="your_api_key_here"
```

Or create:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "your_api_key_here"
```

Never commit this file.

### 6. Start Streamlit

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## ☁️ Deploy to Streamlit Community Cloud

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Sign in with GitHub.
4. Click **Create app**.
5. Select your GitHub repository.
6. Select the branch, normally `main`.
7. Set the main file to:

```text
app.py
```

8. Open **Advanced settings**.
9. Add your secret:

```toml
GROQ_API_KEY = "your_api_key_here"
```

10. Click **Save/Deploy**.
11. Wait for the application to build.
12. Open the generated `streamlit.app` URL.

## 🔐 Security

Never upload your Groq API key to GitHub.

The real `.streamlit/secrets.toml` file should remain local and is ignored by Git.

For Streamlit Community Cloud, put the key in the app's **Secrets** settings.

## ⚠️ Current limitations

### Scanned PDFs

`pypdf` works best when the PDF contains selectable text.

A scanned/image-only PDF may return little or no text. OCR can be added later with tools such as Tesseract or an OCR API.

### Persistence

The FAISS index is created in memory for the current Streamlit session. It is not a permanent cloud database.

This is intentional for this beginner project.

For a production application, consider a persistent vector database such as:

- Qdrant
- Chroma
- Weaviate
- Pinecone
- PostgreSQL + pgvector

### Multiple users

Each Streamlit session builds its own in-memory PDF index. A production multi-user architecture would normally use persistent storage and user/session isolation.

## 🔧 Useful improvements for version 2

After the basic application works, you can add:

- Streaming LLM responses
- Chat history
- Multiple PDF uploads
- PDF document management
- OCR for scanned PDFs
- Hybrid keyword + vector search
- Reranking
- Persistent vector database
- Authentication/login
- Source citations with clickable pages
- Conversation export
- Better chunking based on headings
- Metadata filtering
- Evaluation of retrieval quality

## 📌 Why this is RAG

The application does not send the entire PDF to the LLM.

Instead:

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieve relevant chunks
 ↓
Send relevant context to LLM
 ↓
Answer
```

This is the basic RAG pattern: **Retrieve → Augment → Generate**.

## 📄 License

You can use and modify this project for learning and personal projects.
