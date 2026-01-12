import streamlit as st
from pathlib import Path

from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory
from src.embeddings.build_vectorstore import build_user_vectorstore

USER_ID = "user_001"
UPLOAD_DIR = Path(f"data/uploads/{USER_ID}/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_PATH = Path(f"db/chroma/{USER_ID}")

def load_retriever():
    vectorstore = load_user_vectorstore(USER_ID)
    return get_retriever(vectorstore)

@st.cache_resource
def load_rag_component():
    llm = load_llm()
    memory = ConversationMemory(max_size=5)
    return llm, memory

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

llm, memory = load_rag_component()

st.title("Personal Librarian RAG Chatbot")
st.caption("Ask questions strictly based on the documents uploaded.")

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
retriever = load_retriever()

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
