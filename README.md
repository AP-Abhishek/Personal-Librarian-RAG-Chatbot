# Personal Librarian RAG Chatbot - Intelligent Document QA & Exploration

An intelligent Retrieval-Augmented Generation (RAG) chatbot application built with **Python**, **LangChain**, **HuggingFace Transformers**, **ChromaDB**, and **Streamlit**. It enables users to upload custom PDF documents (research papers, textbooks, notes, novels) and engage in precise, document-grounded conversations with strict refusal guardrails, dual Web UI / CLI interfaces, and full conversation export capabilities.

---

## Project Overview & Application Architecture

The **Personal Librarian RAG Chatbot** leverages an end-to-end open-source AI pipeline to process unstructured documents, construct high-density vector representations, and deliver hallucination-free answers backed by exact source citations.

The core application architecture follows a modular RAG pipeline:

1. **Document Ingestion & Chunking (`src/ingestion/`):** Loads uploaded PDFs via `PyPDFLoader`, injects page and document metadata, and partitions text using `RecursiveCharacterTextSplitter` (1000-char chunks, 200-char overlap).
2. **Embeddings & Vectorstore (`src/embeddings/`):** Generates 384-dimensional vector embeddings using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` and persists isolated per-session collections in `ChromaDB`.
3. **Hybrid Retrieval Engine (`src/retrieval/`):** Features a custom `HybridVectorRetriever` combining top-$k$ semantic vector search with keyword-frequency re-ranking and stopword filtering.
4. **Context Refusal & Generation Chain (`src/generation/`):** Formulates contextual prompts for `google/flan-t5-large`. Evaluates retrieval confidence and enforces strict refusal protocols ("*Not found in the provided documents*") when evidence is insufficient.
5. **Dual Interactive Interfaces (`streamlit_app.py` & `app.py`):** Provides a sleek Streamlit web application alongside an interactive Command Line Interface (CLI) equipped with slash commands (`/upload`, `/list`, `/delete`, `/clean`, `/logs`, `/export`), vague query clarification, and automatic session garbage collection.

---

## Core Modules & Features

* **Strict Document-Grounded QA:** Answers strictly based on uploaded PDFs. Refuses queries with insufficient retrieval confidence or missing evidence.
* **Hybrid Retrieval System:** Combines vector similarity with term-frequency re-ranking to prioritize exact keyword matches alongside semantic meaning.
* **Per-Session Isolation & Privacy:** Each user session operates inside an isolated directory (`db/chroma/session_<uuid>`), ensuring complete multi-tenant data privacy.
* **Dynamic Library Management:** Allows users to add, index, or selectively delete individual PDF files from the active vector store without destroying chat history.
* **Dual Execution Interfaces:** Run either as a modern Streamlit Web Application (`streamlit run streamlit_app.py`) or as a fast Terminal CLI (`python app.py`).
* **CLI Slash Commands:** Full-featured command menu (`/upload`, `/list`, `/delete`, `/clean`, `/logs`, `/verbose`, `/clear`, `/export`, `/exit`) to manage documents and audit logs directly from the terminal.
* **Multi-Format Conversation Export:** Instant single-click export of full chat transcripts with references to `.txt` or `.md`.
* **Automated Session Cleanup:** Background lifecycle daemon and signal handlers (`atexit` / `SIGTERM`) automatically purge session storage directories upon exit or after 2 hours.
* **Conversation Memory & Vague Query Resolution:** Tracks conversation history and asks clarifying follow-ups when vague prompts (e.g., *"tell me more"*) are entered.

---

## Data & Pipeline Details

The system processes document queries through structured pipeline modules:

| Component / Module | Implementation File | Description & Functionality |
| :--- | :--- | :--- |
| **Document Loader** | `src/ingestion/load_pdfs.py` | Extracts text pages from PDFs and injects `file_name`, `page`, and `user_id` metadata. |
| **Text Splitter** | `src/ingestion/chunk_text.py` | Breaks raw document text into overlapping chunks using `RecursiveCharacterTextSplitter`. |
| **Vector Store Manager** | `src/embeddings/build_vectorstore.py` | Caches `all-MiniLM-L6-v2` embeddings, builds Chroma vector databases, and executes per-file vector chunk deletions. |
| **Hybrid Retriever** | `src/retrieval/retriever.py` | Executes hybrid search combining semantic similarity with keyword frequency scoring and stopword filtering. |
| **LLM Generation Pipeline** | `src/generation/llm.py` | Loads local `google/flan-t5-large` model via HuggingFace Transformers pipeline (256 max tokens). |
| **RAG Chain Engine** | `src/generation/rag_chain.py` | Orchestrates query execution, confidence scoring, context trimming, and refusal guardrails. |
| **Conversation Memory** | `src/memory/conversation_memory.py` | Manages query history, vague prompt detection, and disk persistence. |
| **Utility Functions** | `src/utils.py` | Formats multi-turn chat history into exportable Markdown and plain text files. |
| **Streamlit Interface** | `streamlit_app.py` | Main interactive web UI, responsive glassmorphism CSS, session management, and sidebar controls. |
| **CLI Application** | `app.py` | Interactive terminal chatbot interface supporting slash commands, log audit transcripts, and auto-cleanup. |

---

## Tech Stack & Specifications

* **Language:** Python `>=3.10`
* **Orchestration & RAG:** LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`)
* **Embeddings & Vector Database:** `langchain-chroma`, `chromadb`, `langchain-huggingface`, `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Large Language Model:** HuggingFace `transformers` (`google/flan-t5-large`), PyTorch
* **PDF Parsing:** `pypdf`
* **Web Interface:** Streamlit (`>=1.30.0`)

---

## Environment Setup & Running the Application

Follow these steps to run the application locally on your machine:

### Prerequisites
* **Python 3.10** or higher installed.
* **Git** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/AP-Abhishek/Personal-Librarian-RAG-Chatbot.git
cd Personal-Librarian-RAG-Chatbot
```

### Step 2: Create & Activate Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Application

#### Launch Web UI (Streamlit)
```bash
streamlit run streamlit_app.py
```

#### Launch Terminal CLI
```bash
python app.py
```

### CLI Slash Commands Reference

When running `python app.py`, the following interactive commands are available:

```text
  /upload <path>   : Copy & index a PDF file (e.g. /upload "C:\document.pdf")
  /list            : List all indexed PDFs in the active library
  /delete <fname>  : Delete a PDF file by name or path from the library
  /clean           : Wipe all uploaded PDFs, vectorstore & conversation data
  /logs            : Print recorded query log audit transcript
  /verbose         : Toggle verbose RAG logs on/off for future queries
  /clear           : Clear conversation chat history & memory
  /export          : Save chat transcript to cli_chat_history.txt
  /exit            : Quit the chatbot & clean up session files
```

### Step 5: Run Automated Test Suite (Optional)
```bash
pytest
```

---

## Project Structure

```text
Personal-Librarian-RAG-Chatbot/
├── data/                              # Uploaded user PDFs and conversation memory
│   ├── memory/                        # Session conversation history JSON
│   └── uploads/                       # PDF uploads per session
├── db/                                # Persistent Chroma vectorstore databases
│   └── chroma/                        # Per-session SQLite vectorstore collections
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── utils.py                       # Text cleaning, confidence scoring, & export formatters
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── build_vectorstore.py       # Chroma vectorstore creation & per-file deletion
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm.py                     # HuggingFace FLAN-T5-large pipeline loader
│   │   ├── prompt.py                  # Prompt string builder
│   │   └── rag_chain.py               # RAG chain execution, refusal logic, & confidence scoring
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunk_text.py              # Recursive document text chunking
│   │   └── load_pdfs.py               # PyPDFLoader & metadata tagging
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation_memory.py     # Conversation memory & vague query resolution
│   └── retrieval/
│       ├── __init__.py
│       └── retriever.py               # HybridVectorRetriever (Vector + Keyword re-ranking)
├── tests/                             # Automated unit test suite
│   ├── __init__.py
│   ├── embedding_test.py              # Vector store creation tests
│   ├── ingestion_test.py              # PDF parsing and chunking tests
│   ├── rag_test.py                    # RAG chain execution tests
│   └── retriever_test.py              # Hybrid retriever scoring tests
├── app.py                             # Interactive CLI chatbot interface with slash commands
├── streamlit_app.py                   # Main Streamlit Web UI application
├── requirements.txt                   # Project dependency manifest
└── README.md                          # Comprehensive project documentation
```