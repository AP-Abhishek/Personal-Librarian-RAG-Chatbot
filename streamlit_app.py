import streamlit as st

from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory

st.title("Personal Librarian RAG Chatbot")
USER_ID = "user_001"

def load_rag_component():
    vectorstore = load_user_vectorstore(USER_ID)
    retriever = get_retriever(vectorstore)
    llm = load_llm()
    memory = ConversationMemory(max_size=5)
    return retriever, llm, memory

retriever, llm, memory = load_rag_component()

user_query = st.text_input("Ask a question from your documents:")
print(user_query)

if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        memory.add_user_query(user_query)

        if memory.is_vague(user_query):
            last_query = memory.get_last_meaningful_query()
            if not last_query:
                st.warning("Could you clarify what topic you're referring to?")
            else:
                final_query = f"{last_query}. {user_query}"
        else:
            final_query = user_query
        
        print(final_query)
        result = run_rag(llm, retriever, final_query)

        print(result["answer"])
        st.subheader("Answer")
        st.write(result["answer"])

        if result["sources"]:
            print(result["sources"])
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(source)

print("Finished")