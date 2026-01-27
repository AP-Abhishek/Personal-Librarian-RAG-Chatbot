from src.utils import format_export_markdown, format_export_text
import streamlit as st
from pathlib import Path
import shutil, gc, time

from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory
from src.embeddings.build_vectorstore import build_user_vectorstore

st.set_page_config(
    page_title="Personal Librarian",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_USER = "user_001"

with st.sidebar:
    st.title("📚 Librarian")
    st.markdown("---")
    
    st.subheader("👤 Active User")

    if "active_user" not in st.session_state:
        st.session_state.active_user = DEFAULT_USER
    
    new_user = st.text_input(
        "User ID",
        value=st.session_state.active_user
    )

    if st.button("Switch User", use_container_width=True):
        st.session_state.active_user = new_user.strip()
        st.session_state.clear()
        st.rerun()

USER_ID = st.session_state.active_user
UPLOAD_DIR = Path(f"data/uploads/{USER_ID}/pdfs")
VECTORSTORE_PATH = Path(f"db/chroma/{USER_ID}")
MEMORY_PATH = Path(f"data/memory/{USER_ID}/conversation.json")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    
    .stExpander > div:first-child {
        padding: 0.4rem 0.8rem !important;
    }
    
    .stExpander [data-testid="stExpanderDetails"] {
        padding: 0.5rem 1rem 0.75rem 1rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
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

def clear_directory(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

@st.cache_resource
def load_retriever(user_id: str):
    vectorstore = load_user_vectorstore(user_id)
    return get_retriever(vectorstore)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_size=5, persist_path=MEMORY_PATH)

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

def handle_wipe_library():
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.session_state.llm = None
    st.session_state.last_result = None
    st.cache_resource.clear()
    gc.collect()
    time.sleep(0.3)
    clear_directory(UPLOAD_DIR)
    clear_directory(VECTORSTORE_PATH)
    clear_directory(MEMORY_PATH.parent)
    st.session_state.last_action = "Library wiped"
    st.session_state.confirm_wipe = False

with st.sidebar:
    st.subheader("📁 Document Library")
    library_ready = VECTORSTORE_PATH.exists()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{int(library_ready)}"
    )

    if uploaded_files and st.button("Build Library", use_container_width=True):
        st.session_state.is_building = True
        with st.spinner("Analyzing documents..."):
            for uploaded_file in uploaded_files:
                file_path = UPLOAD_DIR / uploaded_file.name
                with open(file_path, "wb") as f:    
                    f.write(uploaded_file.getbuffer())
            build_user_vectorstore(USER_ID)
        st.session_state.is_building = False
        st.session_state.last_action = "Library built"
        st.rerun()

    st.markdown("---")
    st.subheader("📤 Export")

    last_result = st.session_state.get("last_result")

    export_disabled = (
        not last_result 
        or not last_result.get("answer")
        or not last_result.get("sources")
    )

    if last_result:
        st.download_button(
            "Export as Text file",
            data=format_export_text(last_result),
            file_name="librarian_answer.txt",
            mime="text/plain",
            disabled=export_disabled,
            use_container_width=True
        )
        
        st.download_button(
            "Export as Markdown",
            data=format_export_markdown(last_result),
            file_name="librarian_answer.md",
            mime="text/markdown",
            disabled=export_disabled,
            use_container_width=True
        )
    else:
        st.caption("Ask a question to enable export.")
        

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
        st.button(
            "🗑️ Wipe Library",
            type="primary",
            disabled=not confirm_clear,
            use_container_width=True,
            on_click=handle_wipe_library
        )

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
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f'<div class="source-item">{src}</div>', unsafe_allow_html=True)

if st.session_state.is_building:
    st.info("The Librarian is processing your documents. Please stand by...")
    user_query = None
elif not library_ready:
    st.warning("👈 Please upload and build your library in the sidebar to begin.")
    user_query = None
else:
    user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

    memory = st.session_state.memory
    memory.add_user_query(user_query)

    if memory.is_vague(user_query):
        last_query = memory.get_last_meaningful_query()
        if not last_query:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Could you please clarify what topic you're referring to?",
                "sources": []
            })
            st.rerun()
        final_query = f"{last_query}. {user_query}"
    else:
        final_query = user_query

    if st.session_state.llm is None:
        st.session_state.llm = load_llm()

    retriever = load_retriever(USER_ID)

    with st.spinner("Consulting library..."):
        result = run_rag(st.session_state.llm, retriever, final_query)
    
    st.session_state.last_result = {
        "question": final_query,
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"]
    }

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    st.rerun()

