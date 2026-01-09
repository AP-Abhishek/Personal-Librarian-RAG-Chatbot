from src.embeddings.build_vectorstore import build_user_vectorstore

def run_test():
    vs = build_user_vectorstore("user_001")

    results = vs.similarity_search("What is machine learning?", k=3)
    
    for result in results:
        print("----------------------------")
        print(result.page_content)
        print(result.metadata)

if __name__ == "__main__":
    run_test()