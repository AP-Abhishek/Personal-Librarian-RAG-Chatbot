from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory

def main():
    user_id = "user_001"

    vectorstore = load_user_vectorstore(user_id)
    retriever = get_retriever(vectorstore)
    llm = load_llm()
    memory = ConversationMemory()

    print("Personal Librarian RAG Chatbot")
    print("-----------------------------")
    print("Type 'exit' to quit the chatbot")
    print("-----------------------------")
    
    while True:
        user_query = input("User: ")
        if user_query.lower() == "exit":
            break
        
        memory.add_user_query(user_query)
        
        if memory.is_vague(user_query):
            last_query = memory.get_last_meaningful_query()
            if not last_query:
                print("Librarian:")
                print("Could you clarify what topic you're referring to?\n")
                continue

        final_query = user_query
        if memory.is_vague(user_query) and last_query:
            final_query = f"{last_query}. {user_query}"
        
        response = run_rag(llm, retriever, final_query)
        print("Librarian:")
        print(f"Answer -> {response['answer']}")
        print(f"Sources -> {response['sources']}\n")
    
    print(memory.get_history())

if __name__ == "__main__":
    main()
