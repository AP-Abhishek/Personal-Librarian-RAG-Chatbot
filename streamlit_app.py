import streamlit as st
import os
from pathlib import Path

from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory
from src.embeddings.build_vectorstore import build_user_vectorstore

st.title("Personal Librarian RAG Chatbot")
USER_ID = "user_001"
UPLOAD_DIR = Path(f"data/uploads/{USER_ID}/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_PATH = Path(f"db/chroma/{USER_ID}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_rag_component():
    llm = load_llm()
    memory = ConversationMemory(max_size=5)
    return llm, memory

llm, memory = load_rag_component()

st.subheader("Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Build Library"):
        with st.spinner("Processing PDFs and building library..."):
            for uploaded_file in uploaded_files:
                file_path = UPLOAD_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            build_user_vectorstore(USER_ID)
        
        st.success("Library built successfully! You can now ask questions.")

if not VECTORSTORE_PATH.exists():
    st.warning("Please build a library first.")
    st.stop()

def load_retriever():
    vectorstore = load_user_vectorstore(USER_ID)
    return get_retriever(vectorstore)

retriever = load_retriever()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(src)

user_query = st.chat_input("Ask a question from your documents:")
print(user_query)

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
        print(final_query)
        result = run_rag(llm, retriever, final_query)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
    
    st.rerun()
    
print("Finished")