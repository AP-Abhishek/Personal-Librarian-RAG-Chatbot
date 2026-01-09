from src.retrieval.retriever import load_user_vectorstore, get_retriever

def run_test():
    vectorstore = load_user_vectorstore("user_001")

    retriever = get_retriever(
        vectorstore,
        search_type="mmr",
        k=4
    )

    query = "Explain reinforcement learning"
    docs = retriever.invoke(query)

    print(f"Query: {query}")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}")
        print(doc.page_content[:300])
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    run_test()