from rag_app.retrieval import get_retriever

retriever = get_retriever(k=3)

docs = retriever.invoke(
    "What does the document say about responsible gaming?"
)

print("Retrieved documents:", len(docs))

for i, doc in enumerate(docs):
    print(f"\nRESULT {i + 1}")
    print(doc.page_content[:500])
    print("Metadata:", doc.metadata)
    print("-" * 60)