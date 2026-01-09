from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag

def run_test():
    vectorstore = load_user_vectorstore("user_001")
    retriever = get_retriever(vectorstore)
    llm = load_llm()

    queries = [
        "Explain backpropagation",
        "Explain reinforcement learning"
    ]

    for query in queries:
        print("----------------------------------")
        print(f"Query: {query}")

        result = run_rag(llm, retriever, query)

        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")

if __name__ == "__main__":
    run_test()
