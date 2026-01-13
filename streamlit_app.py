import streamlit as st
from pathlib import Path
import shutil

from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory
from src.embeddings.build_vectorstore import build_user_vectorstore

USER_ID = "user_001"
UPLOAD_DIR = Path(f"data/uploads/{USER_ID}/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_PATH = Path(f"db/chroma/{USER_ID}")

def clear_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)

@st.cache_resource
def load_retriever():
    vectorstore = load_user_vectorstore(USER_ID)
    return get_retriever(vectorstore)

@st.cache_resource
def load_rag_component():
    llm = load_llm()
    memory = ConversationMemory(max_size=5)
    return llm, memory

st.set_page_config(
    page_title="Personal Librarian RAG Chatbot",
    page_icon=":books:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    section[data-testid="stChatInput"] {
        margin-top: 2rem;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_building" not in st.session_state:
    st.session_state.is_building = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.llm = None
    st.session_state.memory = ConversationMemory(max_size=5)

if st.session_state.retriever is None:
    try:
        vectorstore = load_user_vectorstore(USER_ID)
        st.session_state.retriever = get_retriever(vectorstore)
    except FileNotFoundError:
        st.session_state.retriever = None

if st.session_state.llm is None:
    st.session_state.llm = load_llm()

if st.session_state.get("delete_library"):
    import gc, time

    st.session_state.retriever = None
    st.session_state.llm = None
    st.cache_resources.clear()
    gc.collect()
    time.sleep(0.5)

    clear_directory(UPLOAD_DIR)
    clear_directory(VECTORSTORE_PATH)

    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.session_state.is_building = False
    st.session_state.delete_library = False

    st.success("Library has been deleted. Please upload new documents to build a new library.")

retriever = st.session_state.retriever
llm = st.session_state.llm
memory = st.session_state.memory

st.title("Personal Librarian RAG Chatbot")
st.caption("Ask questions strictly based on the documents uploaded.")

with st.sidebar:
    st.header("Session Controls")

    if st.button("Reset Chat"):
        st.session_state.chat_history = []
        memory.clear()
        st.success("Chat has been reset.")

    st.divider()

    confirm_clear = st.checkbox("I understand this will delete my library")
    
    if st.button("Clear Library", disabled=not confirm_clear):
        st.session_state.is_building = True
        st.session_state["deleting_library"] = True
        st.rerun()

st.divider()

st.subheader("Your Library")
st.caption("Upload your PDF files to build your library.")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Build Library"):
        st.session_state.is_building = True
        with st.spinner("Processing PDFs and building library..."):
            for uploaded_file in uploaded_files:
                file_path = UPLOAD_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            build_user_vectorstore(USER_ID)
        
        st.session_state.is_building = False
        st.success("Library built successfully! You can now ask questions.")

library_ready = Path(VECTORSTORE_PATH).exists()
retriever = None
if library_ready and not st.session_state.is_building:
    try:
        retriever = load_retriever()
    except Exception:
        retriever = None

st.divider()

st.subheader("Chat with your Library")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(src)

if st.session_state.is_building:
    st.info("Library is being built. Chat will be enabled once it's ready.")
    user_query = None
elif not library_ready:
    st.info("Upload PDFs and build your library to start chatting.")
    user_query = None
else:
    user_query = st.chat_input("Ask a question from your documents")

if user_query:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })

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
        else:
            final_query = f"{last_query}. {user_query}"
    else:
        final_query = user_query
    
    with st.spinner("Searching your documents..."):
        result = run_rag(llm, retriever, final_query)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    st.rerun()
