import uuid

import streamlit as st
from pathlib import Path
import shutil, gc, time

from src.utils import format_chat_export_markdown, format_chat_export_text
from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory
from src.embeddings.build_vectorstore import build_user_vectorstore, delete_pdf_from_user_vectorstore

st.set_page_config(
    page_title="Personal Librarian",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

USER_ID = st.session_state.session_id
UPLOAD_DIR = Path(f"data/uploads/{USER_ID}/pdfs")
VECTORSTORE_PATH = Path(f"db/chroma/{USER_ID}")
MEMORY_PATH = Path(f"data/memory/{USER_ID}/conversation.json")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

def cleanup_stale_sessions(max_age_hours: int = 2, current_session_id: str = ""):
    now = time.time()
    max_age_sec = max_age_hours * 3600

    base_dirs = [Path("db/chroma"), Path("data/uploads"), Path("data/memory")]
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith("session_") and item.name != current_session_id:
                try:
                    mtime = item.stat().st_mtime
                    if now - mtime > max_age_sec:
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass

cleanup_stale_sessions(max_age_hours=2, current_session_id=USER_ID)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 0.25rem !important;
    }

    [data-testid="stVerticalBlock"]:has(> div > .stChatMessage) {
        gap: 0 !important;
    }

    section[data-testid="stSidebar"] .stSubheader {
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stVerticalBlock"] > div:has(div.stChatMessage) {
        max-height: 80vh;
        overflow-y: auto !important;
        padding-bottom: 0.5rem;
    }

    footer {visibility: hidden;}

    .stButton > button {
        border-radius: 8px !important;
        transition: all 0.1s ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        margin-bottom: 0.75rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-0.5px);
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
    }

    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        background: rgba(255, 255, 255, 0.01) !important;
        border-radius: 8px !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
    }

    .stExpander summary p {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    .stExpander > div:first-child {
        padding: 0.4rem 0.8rem !important;
    }

    .stExpander [data-testid="stExpanderDetails"] {
        padding: 0.5rem 0.6rem 0.75rem 0.6rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
    }

    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 6px !important;
        padding: 0.35rem 0.5rem !important;
        margin-bottom: 0.4rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stHorizontalBlock"] button {
        background: rgba(239, 68, 68, 0.12) !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
        color: #f87171 !important;
        border-radius: 6px !important;
        padding: 0 !important;
        margin: 0 !important;
        height: 28px !important;
        width: 28px !important;
        min-height: 28px !important;
        min-width: 28px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
        transform: none !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stHorizontalBlock"] button:hover {
        background: rgba(239, 68, 68, 0.3) !important;
        border-color: rgba(239, 68, 68, 0.6) !important;
        color: #ffffff !important;
    }

    .source-item {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
        line-height: 1.5;
        display: block;
    }

    .assistant-answer {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def wipe_vectorstore(user_id: str):
    try:
        load_retriever.clear()
        gc.collect()
        time.sleep(0.1)
        vs_dir = Path(f"db/chroma/{user_id}")
        if vs_dir.exists():
            shutil.rmtree(vs_dir, ignore_errors=True)
        return True
    except Exception:
        return False

@st.cache_resource
def load_retriever(user_id: str):
    vectorstore = load_user_vectorstore(user_id)
    return get_retriever(vectorstore)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_size=5, persist_path=MEMORY_PATH)
    if not st.session_state.chat_history:
        st.session_state.memory.clear()

if "llm" not in st.session_state:
    st.session_state.llm = None

if "is_building" not in st.session_state:
    st.session_state.is_building = False

if "confirm_wipe" not in st.session_state:
    st.session_state.confirm_wipe = False

if "last_action" not in st.session_state:
    st.session_state.last_action = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "pending_wipe" not in st.session_state:
    st.session_state.pending_wipe = False

if st.session_state.pending_wipe:
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.session_state.llm = None
    st.session_state.last_result = None

    load_retriever.clear()
    gc.collect()
    time.sleep(0.1)

    wiped = wipe_vectorstore(USER_ID)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(MEMORY_PATH.parent, ignore_errors=True)

    st.session_state.pending_wipe = False
    st.session_state.last_action = "Session library wiped" if wiped else "No library found"
    st.rerun()

with st.sidebar:
    st.title("📚 Librarian")
    st.markdown("---")
    st.subheader("📁 Document Library")
    existing_pdfs = list(UPLOAD_DIR.glob("*.pdf")) if UPLOAD_DIR.exists() else []
    library_ready = VECTORSTORE_PATH.exists() and bool(existing_pdfs)

    if library_ready and existing_pdfs:
        st.caption("🟢 **Library Status:** Active")
        with st.expander(f"📚 Indexed Files ({len(existing_pdfs)})", expanded=True):
            for pdf in existing_pdfs:
                file_size_kb = pdf.stat().st_size / 1024
                c1, c2 = st.columns([0.82, 0.18])
                with c1:
                    st.markdown(
                        f"<div style='font-size:0.83rem; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;' title='{pdf.name}'>📄 {pdf.name}</div>"
                        f"<div style='font-size:0.72rem; color:#94a3b8;'>Size: {file_size_kb:.1f} KB</div>",
                        unsafe_allow_html=True
                    )
                with c2:
                    if st.button("🗑️", key=f"del_{pdf.name}", help=f"Delete {pdf.name}"):
                        pdf_name = pdf.name
                        delete_pdf_from_user_vectorstore(USER_ID, pdf_name)
                        pdf.unlink(missing_ok=True)
                        remaining = list(UPLOAD_DIR.glob("*.pdf"))
                        load_retriever.clear()
                        st.session_state.llm = None
                        if remaining:
                            st.session_state.last_action = f"Deleted {pdf_name}"
                        else:
                            wipe_vectorstore(USER_ID)
                            st.session_state.last_action = f"Deleted {pdf_name} (Library empty)"
                        st.rerun()
        
        with st.expander("➕ Add Documents"):
            uploaded_files = st.file_uploader(
                "Upload additional PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="uploader_add"
            )
            if uploaded_files and st.button("Update Library", use_container_width=True):
                st.session_state.is_building = True
                try:
                    with st.spinner("Analyzing documents..."):
                        for uploaded_file in uploaded_files:
                            file_path = UPLOAD_DIR / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        build_user_vectorstore(USER_ID)
                        load_retriever.clear()
                        st.session_state.llm = None
                    st.session_state.last_action = "Library updated successfully"
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    st.session_state.last_action = "Failed to update library"
                finally:
                    st.session_state.is_building = False
                    st.rerun()
    else:
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader_initial"
        )

        if uploaded_files and st.button("Build Library", use_container_width=True):
            st.session_state.is_building = True
            try:
                with st.spinner("Analyzing documents..."):
                    for uploaded_file in uploaded_files:
                        file_path = UPLOAD_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    build_user_vectorstore(USER_ID)
                    load_retriever.clear()
                    st.session_state.llm = None
                st.session_state.last_action = "Library built successfully"
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                st.session_state.last_action = "Failed to build library"
            finally:
                st.session_state.is_building = False
                st.rerun()

    st.markdown("---")
    st.subheader("📤 Export")

    chat_history = st.session_state.get("chat_history", [])
    export_disabled = not chat_history

    st.download_button(
        "Export as Text file",
        data=format_chat_export_text(chat_history) if chat_history else "",
        file_name="chat_history.txt",
        mime="text/plain",
        disabled=export_disabled,
        use_container_width=True
    )

    st.download_button(
        "Export as Markdown",
        data=format_chat_export_markdown(chat_history) if chat_history else "",
        file_name="chat_history.md",
        mime="text/markdown",
        disabled=export_disabled,
        use_container_width=True
    )

    if export_disabled:
        st.caption("Start a conversation to enable export.")

    st.markdown("---")
    st.subheader("⚙️ Controls")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.session_state.last_action = "Chat cleared"
        st.rerun()

    with st.expander("⚠️ Danger Zone"):
        confirm_clear = st.checkbox(
            "Confirm permanent deletion",
            key="confirm_wipe"
        )
        if st.button(
            "🗑️ Wipe Library",
            type="primary",
            disabled=not confirm_clear,
            use_container_width=True
        ):
            st.session_state.pending_wipe = True

if st.session_state.last_action:
    st.toast(st.session_state.last_action)
    st.session_state.last_action = None

st.title("Personal Librarian Chat")
st.caption("Intelligent document retrieval and exploration.")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources & References"):
                    for src in msg["sources"]:
                        if isinstance(src, dict):
                            pdf = src.get("pdf", "Document")
                            page = src.get("page", "1")
                            snippet = src.get("snippet", "")
                            st.markdown(f'<div class="source-item">📄 <b>{pdf}</b> (Page {page})<br/><i>"{snippet}"</i></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="source-item">{src}</div>', unsafe_allow_html=True)

if st.session_state.is_building:
    st.info("The Librarian is processing your documents. Please stand by...")
    user_query = None
elif not library_ready:
    if not st.session_state.chat_history:
        st.warning("👈 Please upload and build your library in the sidebar to begin.")
    else:
        st.info("💡 Library is currently empty. Upload documents to continue asking questions, or export your conversation in the sidebar.")
    user_query = None
else:
    user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    memory = st.session_state.memory

    if memory.is_vague(user_query):
        last_query = memory.get_last_meaningful_query()
        if not last_query:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Could you please clarify what specific topic or document section you're referring to?",
                "sources": []
            })
            memory.add_user_query(user_query)
            st.rerun()
        final_query = f"{last_query}. {user_query}"
    else:
        final_query = user_query

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    memory.add_user_query(user_query)

    if st.session_state.llm is None:
        st.session_state.llm = load_llm()

    retriever = load_retriever(USER_ID)

    try:
        with st.spinner("Consulting library..."):
            result = run_rag(st.session_state.llm, retriever, final_query)
    except Exception as e:
        logging.error(f"Error executing RAG chain: {e}", exc_info=True)
        result = {
            "answer": f"An error occurred while consulting the library: {str(e)}",
            "sources": [],
            "confidence": 0.0
        }

    st.session_state.last_result = {
        "question": final_query,
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result.get("confidence", 0.0)
    }

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    st.rerun()
